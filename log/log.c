/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "util/log.h"
#include "util/buffer.h"
#include "comm/easy.h"


/** 日志接口。*/
typedef struct _abcdk_log
{
    /** 环境初始化状态。*/
    volatile int init_status;

    /** 标识。*/
    char *ident;

    /** 收货人。*/
    char *consignee;

    /** 掩码。*/
    volatile uint32_t mask;

    /** 通讯。*/
    abcdk_comm_easy_t *easy;
        
    /** 缓存的线程KEY。*/
    pthread_key_t buf_ptkey;

}abcdk_log_t;

/** 
 * 日志缓存。
 * 
 * @note 线程之间是独立的。
*/
typedef struct _abcdk_log_buffer
{
    abcdk_buffer_t *buf;

}abcdk_log_buffer_t;

/**
 * 初始化参数。
 */
typedef struct _abcdk_log_init_params
{
    abcdk_log_t *ctx;
    const char *ident;
    const char *consignee;
} abcdk_log_init_params_t;

void _abcdk_log_buf_destroy(void *opaque)
{
    abcdk_log_buffer_t *buf_p = (abcdk_log_buffer_t*)opaque;

    /*释放对象内部成员。*/
    abcdk_buffer_free(&buf_p->buf);

    /*释放对象本身。*/
    abcdk_heap_free2((void**)&buf_p);
}

int _abcdk_log_init(void *opaque)
{
    abcdk_log_init_params_t *params = (abcdk_log_init_params_t *)opaque;

    params->ctx->ident = abcdk_strdup(params->ident);
    params->ctx->consignee = abcdk_strdup(params->consignee);
    params->ctx->mask = 0xFFFFFFFF;
    params->ctx->easy = NULL;
    pthread_key_create(&params->ctx->buf_ptkey,_abcdk_log_buf_destroy);



    return 0;
}

abcdk_log_t *_abcdk_log_ctx()
{
    static abcdk_log_t ctx = {0};
    int chk;

    chk = abcdk_once(&ctx.init_status, _abcdk_log_init, &ctx);
    assert(chk >= 0);

    return &ctx;
}

void abcdk_log_open(const char *ident,const char *consignee)
{

}

void abcdk_log_mask(uint32_t mask)
{
    
}

void abcdk_log_vprintf(int priority, const char *fmt, va_list ap)
{
    char buf[4096] = {0};

    assert(fmt);

    /*获取线程名称。*/
    abcdk_thread_getname(buf);

    /*如果线程已命名，则拼接在行首。*/
    snprintf(buf + strlen(buf), 4080, "%s%s", (*buf ? ": " : ""), fmt);

}

void abcdk_log_printf(int priority, const char *fmt, ...)
{
    assert(fmt);

    va_list ap;
    va_start(ap, fmt);
    abcdk_log_vprintf(priority,fmt, ap);
    va_end(ap);
}