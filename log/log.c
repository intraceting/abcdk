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
 * -----------------------------------------------------------------------------------
 * |Message Data                                                                     |
 * -----------------------------------------------------------------------------------
 * |Microsecond  |Service ID |Process ID |Name      |Reserve  |Cargo Length |Cargo   |
 * |8 Bytes      |2 Bytes    |4 Bytes    |16 Bytes  |1 Bytes  |4 Bytes      |N Bytes |
 * -----------------------------------------------------------------------------------
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
    abcdk_comm_node_t *easy;

    /** 通讯链路锁。*/
    abcdk_mutex_t easy_mutex;

    /** 服务编号。*/
    uint16_t service;

    /** 收货人。*/
    const char *consignee;

    /** 是否复制到syslog。!0 是，0 否。*/
    volatile int copy2syslog;

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
    ctx->service = 1;
    ctx->copy2syslog = 0;
    ctx->consignee = NULL;
    ctx->mask = 0xFFFFFFFF;
    pthread_key_create(&ctx->ptkey, _abcdk_log_buf_destroy);
    abcdk_mutex_init2(&ctx->easy_mutex,0);

    cons_p = getenv("ABCDK_LOG_CONSIGNEE");
    if (cons_p && *cons_p)
    {
        ctx->consignee = abcdk_strdup(cons_p);
        ctx->comm = abcdk_comm_start(1,-1);
    }

    return 0;
}

void _abcdk_log_uninit()
{
    abcdk_log_t *ctx = _abcdk_log_ctx();

    abcdk_comm_stop(&ctx->comm);
    abcdk_comm_unref(&ctx->easy);
    pthread_key_delete(ctx->ptkey);
    abcdk_mutex_unlock(&ctx->easy_mutex);
    abcdk_heap_free2((void**)&ctx->consignee);
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

void abcdk_log_open(const char *consignee,uint16_t service, int copy2syslog)
{
    abcdk_log_t *ctx = NULL;

    ABCDK_ASSERT(service > 0 && service <= 65535, "service在1~65535之间有效。");

    /*设置环境变量，具体的初始化，按需执行一次。*/
    if (consignee)
        setenv("ABCDK_LOG_CONSIGNEE", consignee, 1);

    ctx = _abcdk_log_get_ctx();
    ctx->service = service;
    ctx->copy2syslog = copy2syslog;
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

void _abcdk_log_easy_request_cb(abcdk_comm_node_t *easy, const void *req, size_t len)
{
    char sockname[NAME_MAX] = {0}, peername[NAME_MAX] = {0};
    
    if(easy)
        abcdk_comm_get_sockaddr_str(easy,sockname,peername);

    // if(!req)
    //     fprintf(stderr,"Disconnected(%s -> %s).\n",sockname, peername);

}

abcdk_comm_node_t *_abcdk_log_get_easy()
{
    abcdk_log_t *ctx = _abcdk_log_get_ctx();
    abcdk_sockaddr_t addr = {0};
    abcdk_comm_node_t *easy_p = NULL;
    int chk;

    abcdk_mutex_lock(&ctx->easy_mutex,1);

    if (!ctx->easy || abcdk_comm_easy_state(ctx->easy) != 0)
    {
        /*释放已经断开的。*/
        abcdk_comm_unref(&ctx->easy);
        
        /*指定收货人再尝试连接，否则没意义。*/
        if (ctx->consignee)
        {
            abcdk_sockaddr_from_string(&addr, ctx->consignee, 1);
            ctx->easy = abcdk_comm_easy_alloc(ctx->comm,666666666);

            abcdk_comm_easy_callback_t cb = {NULL,_abcdk_log_easy_request_cb};
            chk = abcdk_comm_easy_connect(ctx->easy, NULL, &addr, &cb);
            if (chk != 0)
                abcdk_comm_unref(&ctx->easy);
        }
    }

    if(ctx->easy)
        easy_p = abcdk_comm_refer(ctx->easy);

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

int _abcdk_log_send(const void *data, size_t len)
{
    abcdk_log_t *ctx = _abcdk_log_get_ctx();
    abcdk_comm_node_t *easy_p = NULL;
    int chk;

    /*获取通讯链路。*/
    easy_p = _abcdk_log_get_easy();
    if (!easy_p)
        return -2;

    /*发送到远程。*/
    chk = abcdk_comm_easy_request(easy_p, data, len, NULL);
    abcdk_comm_unref(&easy_p);

    return chk;
}

void abcdk_log_vprintf(int type, const char *fmt, va_list ap)
{
    abcdk_log_t *ctx = _abcdk_log_get_ctx();
    abcdk_object_t *buf_p = NULL;
    uint64_t ts = 0;
    char name[17] = {0};
    uint32_t len = 0;
    int chk;

    assert(fmt != NULL);

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
    vsnprintf(ABCDK_PTR2I8PTR(buf_p->pptrs[0], 35), buf_p->sizes[0] - 35, fmt, ap);
    len = strlen(ABCDK_PTR2I8PTR(buf_p->pptrs[0], 35));
    /*填充其它信息。*/
    ABCDK_PTR2I64(buf_p->pptrs[0], 0) = abcdk_endian_h_to_b64(ts);
    ABCDK_PTR2I16(buf_p->pptrs[0], 8) = abcdk_endian_h_to_b16(ctx->service);
    ABCDK_PTR2I32(buf_p->pptrs[0], 10) = abcdk_endian_h_to_b32(getpid());
    strncpy(ABCDK_PTR2I8PTR(buf_p->pptrs[0], 14), name, 16);
    ABCDK_PTR2I8(buf_p->pptrs[0], 30) = 0;
    ABCDK_PTR2I32(buf_p->pptrs[0], 31) = abcdk_endian_h_to_b32(len);

    /*可能需要复制到syslog。*/
    if (ctx->copy2syslog)
        syslog(type, "s%hu.p%d %s: %s\n", ctx->service, getpid(), name, ABCDK_PTR2I8PTR(buf_p->pptrs[0], 35));

    /*发送到远程。因连接有可能被断开，尝试重发一次。*/
    chk = _abcdk_log_send(buf_p->pptrs[0], 35 + len);
    if (chk != 0)
        chk = _abcdk_log_send(buf_p->pptrs[0], 35 + len);
}

void abcdk_log_printf(int type, const char *fmt, ...)
{
    assert(fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    abcdk_log_vprintf(type, fmt, ap);
    va_end(ap);
}