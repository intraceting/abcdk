/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "log/log.h"
#include "util/object.h"
#include "util/uri.h"
#include "shell/proc.h"
#include "comm/easy.h"

/*
 * -----------------------------------------------------------
 * |Message Data                                             |
 * -----------------------------------------------------------
 * |Microsecond  |Name      |Reserve  |Cargo Length |Cargo   |
 * |8 Bytes      |16 Bytes  |1 Bytes  |4 Bytes      |N Bytes |
 * -----------------------------------------------------------
*/

/** 日志接口。*/
typedef struct _abcdk_log
{
    /** 环境初始化状态。*/
    volatile int init_status;

    /** 线程KEY。*/
    pthread_key_t ptkey;

    /** 通讯环境。*/
    abcdk_comm_t *comm;

    /** 通讯链路。*/
    abcdk_comm_easy_t *easy;

    /** 通讯链路锁。*/
    abcdk_mutex_t easy_mutex;

    /** 收货人。*/
    char consignee[NAME_MAX];

    /** 掩码。*/
    volatile uint32_t mask;

} abcdk_log_t;

abcdk_log_t *_abcdk_log_ctx()
{
    static abcdk_log_t ctx = {0};
    return &ctx;
}

void _abcdk_log_buf_destroy(void *opaque)
{
    abcdk_object_t *buf_p = (abcdk_object_t *)opaque;
    abcdk_object_unref(&buf_p);
}

int _abcdk_log_init(void *opaque)
{
    abcdk_log_t *ctx = (abcdk_log_t *)opaque;
    char *cons_p = NULL;

    ctx->comm = NULL;
    ctx->easy = NULL;
    ctx->mask = 0xFFFFFFFF;
    pthread_key_create(&ctx->ptkey, _abcdk_log_buf_destroy);
    abcdk_mutex_init2(&ctx->easy_mutex,0);

    cons_p = getenv(ABCDK_LOG_CONSIGNEE);
    if (cons_p && *cons_p)
        strncpy(ctx->consignee,cons_p,NAME_MAX);
    else
        strncpy(ctx->consignee,"127.0.0.1:65535",NAME_MAX);

    ctx->comm = abcdk_comm_start(1);

    return 0;
}

void _abcdk_log_uninit()
{
    abcdk_log_t *ctx = _abcdk_log_ctx();

    abcdk_comm_stop(&ctx->comm);
    abcdk_comm_easy_unref(&ctx->easy);
    pthread_key_delete(ctx->ptkey);
    abcdk_mutex_unlock(&ctx->easy_mutex);
}

abcdk_log_t *_abcdk_log_get_ctx()
{
    abcdk_log_t *ctx = _abcdk_log_ctx();
    int chk;

    chk = abcdk_once(&ctx->init_status, _abcdk_log_init, ctx);
    assert(chk >= 0);

    /*第一次初始化，要注册反初始化函数。*/
    if (chk == 0)
        atexit(_abcdk_log_uninit);

    return ctx;
}

void abcdk_log_close()
{
    _abcdk_log_uninit();
}

void abcdk_log_open(const char *consignee)
{
    /*设置环境变量，具体的初始，按需执行一次。*/
    if (consignee)
        setenv(ABCDK_LOG_CONSIGNEE, consignee, 1);
}

void abcdk_log_mask(int type, ...)
{
    abcdk_log_t *ctx = _abcdk_log_get_ctx();
    uint32_t mask = 0;

    va_list vaptr;
    va_start(vaptr, type);
    for (;;)
    {
        if (type < ABCDK_LOG_ERROR || type >= ABCDK_LOG_MAX)
            break;

        mask |= (1 << type);

        /*遍历后续的。*/
        type = va_arg(vaptr, int);
    }
    va_end(vaptr);

    /*覆盖现有的。*/
    abcdk_atomic_store(&ctx->mask, mask);
}

void _abcdk_log_easy_request_cb(abcdk_comm_easy_t *easy, const void *req, size_t len)
{
    char sockname[NAME_MAX] = {0}, peername[NAME_MAX] = {0};
    
    if(easy)
        abcdk_comm_easy_get_sockaddr_str(easy,sockname,peername);

    if(!req)
        fprintf(stderr,"Disconnected(%s -> %s).\n",sockname, peername);

}

abcdk_comm_easy_t *_abcdk_log_get_easy()
{
    abcdk_log_t *ctx = _abcdk_log_get_ctx();
    abcdk_sockaddr_t addr = {0};
    abcdk_comm_easy_t *easy_p = NULL;

    abcdk_mutex_lock(&ctx->easy_mutex,1);

    if (!ctx->easy || abcdk_comm_easy_state(ctx->easy) != 0)
    {
        abcdk_comm_easy_unref(&ctx->easy);
        abcdk_sockaddr_from_string(&addr, ctx->consignee, 1);
        ctx->easy = abcdk_comm_easy_connect(ctx->comm, NULL, &addr, _abcdk_log_easy_request_cb, NULL);
        if (ctx->easy)
            abcdk_comm_easy_set_timeout(ctx->easy, -1);
    }

    if(ctx->easy)
        easy_p = abcdk_comm_easy_refer(ctx->easy);

    abcdk_mutex_unlock(&ctx->easy_mutex);

    return easy_p;
}

abcdk_object_t *_abcdk_log_get_buffer()
{
    abcdk_log_t *ctx = _abcdk_log_get_ctx();
    abcdk_object_t *buf_p = NULL;

    buf_p = pthread_getspecific(ctx->ptkey);
    if (!buf_p)
    {
        /*没有缓存，申请一个。*/
        buf_p = abcdk_object_alloc2(16 * 1024 * 1024);
        if (buf_p)
            pthread_setspecific(ctx->ptkey, buf_p);
    }

    return buf_p;
}

void abcdk_log_vprintf(int type, const char *fmt, va_list ap)
{
    abcdk_log_t *ctx = _abcdk_log_get_ctx();
    abcdk_object_t *buf_p = NULL;
    abcdk_comm_easy_t *easy_p = NULL;
    uint64_t ts = 0;
    char name[17] = {0};
    uint32_t len = 0;
    int chk;

    assert(fmt != NULL && ap != NULL);

    /*不知道什么类型，直接跳过。*/
    if (type < ABCDK_LOG_ERROR || type >= ABCDK_LOG_MAX)
        return;

    /*如果不需要记录，直接跳过。*/
    if (!(abcdk_atomic_load(&ctx->mask) & (1 << type)))
        return;

    /*获取自然时间。*/
    ts = abcdk_time_clock2kind_with(CLOCK_REALTIME, 6);
    /*获取线程名称。*/
    abcdk_thread_getname(name);
    
    /*获取缓存。*/
    buf_p = _abcdk_log_get_buffer();
    if (!buf_p)
        return;

    /*格式化货物。*/
    vsnprintf(ABCDK_PTR2I8PTR(buf_p->pptrs[0], 29), buf_p->sizes[0] - 29, fmt, ap);
    len = strlen(ABCDK_PTR2I8PTR(buf_p->pptrs[0], 29));
    /*填充其它信息。*/
    ABCDK_PTR2I64(buf_p->pptrs[0], 0) = abcdk_endian_h_to_b64(ts);
    strncpy(ABCDK_PTR2I8PTR(buf_p->pptrs[0], 8), name, 16);
    ABCDK_PTR2I8(buf_p->pptrs[0], 24) = 0;
    ABCDK_PTR2I32(buf_p->pptrs[0], 25) = abcdk_endian_h_to_b32(len);

    /*获取通讯链路。*/
    easy_p = _abcdk_log_get_easy();
    if (!easy_p)
        return;

    /*发送到远程。*/
    abcdk_comm_easy_request(easy_p, buf_p->pptrs[0], 29 + len, NULL);
    abcdk_comm_easy_unref(&easy_p);
}

void abcdk_log_printf(int type, const char *fmt, ...)
{
    assert(fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    abcdk_log_vprintf(type, fmt, ap);
    va_end(ap);
}