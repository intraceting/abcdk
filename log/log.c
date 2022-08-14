/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "log/log.h"
#include "util/object.h"
#include "shell/proc.h"
#include "comm/easy.h"

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

    /** 通讯状态。*/
    volatile int easy_state;

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

    cons_p = getenv(ABCDK_LOG_CONSIGNEE);
    if (cons_p && *cons_p)
        strncpy(ctx->consignee, cons_p, NAME_MAX);
    else
        strncpy(ctx->consignee, "127.0.0.1:65535", NAME_MAX);

    ctx->comm = abcdk_comm_start(1);

    return 0;
}

void _abcdk_log_uninit()
{
    abcdk_log_t *ctx = _abcdk_log_ctx();

    abcdk_comm_stop(&ctx->comm);
    abcdk_comm_easy_unref(&ctx->easy);
    pthread_key_delete(ctx->ptkey);
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

void abcdk_log_vprintf(int type, const char *fmt, va_list ap)
{
    abcdk_log_t *ctx = _abcdk_log_get_ctx();
    abcdk_object_t *buf_p = NULL;
    struct timespec ts = {0};
    char name[17] = {0};
    struct tm tm = {0};
    int prefix_len = 0;

    assert(fmt != NULL && ap != NULL);

    /*不知道什么类型，直接跳过。*/
    if (type < ABCDK_LOG_ERROR || type >= ABCDK_LOG_MAX)
        return;

    /*如果不需要记录，直接跳过。*/
    if (!(abcdk_atomic_load(&ctx->mask) & (1 << type)))
        return;

    /*获取自然时间。*/
    clock_gettime(CLOCK_REALTIME, &ts);
    /*获取线程名称。*/
    abcdk_thread_getname(name);
    /*秒转本地时间。*/
    abcdk_time_sec2tm(&tm, ts.tv_sec, 0);

    /*获取缓存。*/
    buf_p = pthread_getspecific(ctx->ptkey);
    if (!buf_p)
    {
        /*创建缓存。*/
        buf_p = abcdk_object_alloc2(16 * 1024 * 1024);
        if (buf_p)
            pthread_setspecific(ctx->ptkey, buf_p);
    }

    /*格式化前缀。*/
    snprintf(buf_p->pptrs[0], buf_p->sizes[0], "%d-%02d-%02d %02d:%02d:%02d [%d] %s: ",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour + 1, tm.tm_min, tm.tm_sec,
             getpid(), name);

    /*拼接日志。*/
    prefix_len = strlen((char *)buf_p->pptrs[0]);
    vsnprintf(ABCDK_PTR2I8PTR(buf_p->pptrs[0], prefix_len), buf_p->sizes[0] - prefix_len, fmt, ap);

    fprintf(stderr, "%s\n", (char *)buf_p->pptrs[0]);
}

void abcdk_log_printf(int type, const char *fmt, ...)
{
    assert(fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    abcdk_log_vprintf(type, fmt, ap);
    va_end(ap);
}