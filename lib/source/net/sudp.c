/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/net/sudp.h"

/**简单的UDP环境。 */
struct _abcdk_sudp
{
    /**配置。*/
    abcdk_sudp_config_t cfg;

    /**线程池配置。*/
    abcdk_worker_config_t worker_cfg;

    /**线程池环境。*/
    abcdk_worker_t *worker_ctx;

    /**线程池标志。0： 运行，!0：停止。*/
    volatile int worker_flag;

    /** 索引。*/
    uint64_t index;

    /**句柄。*/
    int fd;

    /**密钥状态。0 禁用, !0 启用。*/
    volatile int cipher_ok;

    /**密钥同步锁。*/
    abcdk_rwlock_t *cipher_locker;

#ifdef OPENSSL_VERSION_NUMBER
    /**密钥环境。*/
    abcdk_openssl_cipherex_t *cipherex_out;
    abcdk_openssl_cipherex_t *cipherex_in;
#endif //OPENSSL_VERSION_NUMBER
} ;//abcdk_sudp_t;

void abcdk_sudp_destroy(abcdk_sudp_t **ctx)
{
    abcdk_sudp_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    ABCDK_ASSERT(ctx_p->worker_ctx == NULL, "销毁前必须先停止。");

#ifdef OPENSSL_VERSION_NUMBER
    abcdk_openssl_cipherex_destroy(&ctx_p->cipherex_in);
    abcdk_openssl_cipherex_destroy(&ctx_p->cipherex_out);
#endif //OPENSSL_VERSION_NUMBER

    abcdk_rwlock_destroy(&ctx_p->cipher_locker);
    abcdk_closep(&ctx_p->fd);
    
    abcdk_heap_free(ctx_p);
}

static void _abcdk_sudp_process_input_cb(void *opaque,uint64_t event,void *item);

static void _abcdk_sudp_fix_cfg(abcdk_sudp_t *ctx)
{
    /*修复不支持的配置和默认值。*/
    if (ctx->cfg.input_worker <= 0)
        ctx->cfg.input_worker = abcdk_align(sysconf(_SC_NPROCESSORS_ONLN) / 2, 2);

}
abcdk_sudp_t *abcdk_sudp_create(abcdk_sudp_config_t *cfg)
{
    abcdk_sudp_t *ctx;
    int sock_flag;
    int have_port;
    int chk;

    assert(cfg != NULL);
    assert(cfg->listen_addr.family == AF_INET ||cfg->listen_addr.family == AF_INET6);
    assert(cfg->input_cb != NULL);

    ctx = (abcdk_sudp_t*)abcdk_heap_alloc(sizeof(abcdk_sudp_t));
    if(!ctx)
        return NULL;

    ctx->cfg = *cfg;

    _abcdk_sudp_fix_cfg(ctx);

    ctx->index = abcdk_sequence_num();
    ctx->fd = -1;

    ctx->fd = abcdk_socket(ctx->cfg.listen_addr.family,1);
    if(ctx->fd < 0)
        goto ERR;

    /*端口复用，用于快速重启恢复。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(ctx->fd, SOL_SOCKET, SO_REUSEPORT, &sock_flag, 2);
    if (chk != 0)
        goto ERR;

    /*地址复用，用于快速重启恢复。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(ctx->fd, SOL_SOCKET, SO_REUSEADDR, &sock_flag, 2);
    if (chk != 0)
        goto ERR;

    if(ctx->cfg.listen_addr.family == AF_INET6)
    {
        /*IPv6仅支持IPv6。*/
        sock_flag = 1;
        chk = abcdk_sockopt_option_int(ctx->fd, IPPROTO_IPV6, IPV6_V6ONLY, &sock_flag, 2);
        if (chk != 0)
            goto ERR;
    }

    if(ctx->cfg.listen_addr.family == AF_INET)
        have_port = (ctx->cfg.listen_addr.addr4.sin_port != 0);
    else if(ctx->cfg.listen_addr.family == AF_INET6)
        have_port = (ctx->cfg.listen_addr.addr6.sin6_port != 0);

    if (have_port)
    {
        chk = abcdk_bind(ctx->fd, &ctx->cfg.listen_addr);
        if (chk != 0)
            goto ERR;

        if(ctx->cfg.mreq_enable)
        {
            chk = abcdk_socket_option_multicast(ctx->fd,ctx->cfg.listen_addr.family,&ctx->cfg.mreq_addr,1);
            if (chk != 0)
                goto ERR;
        }
    }
    else if(ctx->cfg.mreq_enable)
    {
        abcdk_trace_output(LOG_WARNING,"未绑定端口，忽略组播配置。");
    }

    chk = abcdk_fflag_add(ctx->fd,O_NONBLOCK);
    if(chk != 0)
        goto ERR;

    ctx->cipher_locker = abcdk_rwlock_create();
    if(!ctx->cipher_locker)
        goto ERR;

    ctx->worker_cfg.numbers = ctx->cfg.input_worker; 
    ctx->worker_cfg.opaque = ctx;
    ctx->worker_cfg.process_cb = _abcdk_sudp_process_input_cb;

    ctx->worker_ctx = abcdk_worker_start(&ctx->worker_cfg);
    if(!ctx->worker_ctx)
        goto ERR;

    for(int i = 0;i<ctx->worker_cfg.numbers;i++)
        abcdk_worker_dispatch(ctx->worker_ctx, 0x01, (void *)0);

    return ctx;

ERR:

    abcdk_sudp_destroy(&ctx);

    return NULL;
}

void abcdk_sudp_stop(abcdk_sudp_t *ctx)
{
    if(!ctx)
        return;
    
    /*通知线程退出。*/
    abcdk_atomic_store(&ctx->worker_flag,1);

    /*等待线程结束。*/
    abcdk_worker_stop(&ctx->worker_ctx);
}

int abcdk_sudp_cipher_reset(abcdk_sudp_t *ctx,const uint8_t *key,size_t klen,int flag)
{
    int chk = -1;

    assert(ctx != NULL && key != NULL && klen > 0);

    abcdk_rwlock_wrlock(ctx->cipher_locker,1);

#ifdef OPENSSL_VERSION_NUMBER

    if(flag & 0x01)
    {
        /*关闭旧的，并创建新的。*/
        abcdk_openssl_cipherex_destroy(&ctx->cipherex_in);
        ctx->cipherex_in = abcdk_openssl_cipherex_create(4,ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM,key,klen);
    }

    if(flag & 0x02)
    {
        /*关闭旧的，并创建新的。*/
        abcdk_openssl_cipherex_destroy(&ctx->cipherex_out);
        ctx->cipherex_out = abcdk_openssl_cipherex_create(4,ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM,key,klen);
    }

    /*必须都成功。*/
    chk = ((ctx->cipherex_in && ctx->cipherex_out) ? 0 : -1);

#else //OPENSSL_VERSION_NUMBER
    abcdk_trace_output(LOG_WARNING, "当前环境未包含加密套件，忽略密钥文件。");
#endif //OPENSSL_VERSION_NUMBER

    abcdk_rwlock_unlock(ctx->cipher_locker);

    return chk;
}

static abcdk_object_t *_abcdk_sudp_cipher_update_pack(abcdk_sudp_t *ctx,void *in, int in_len,int enc)
{
    abcdk_object_t *dst_p = NULL;

    abcdk_rwlock_rdlock(ctx->cipher_locker,1);

#ifdef OPENSSL_VERSION_NUMBER
    if(enc)
    {
        if(ctx->cipherex_out)
            dst_p = abcdk_openssl_cipherex_update_pack(ctx->cipherex_out,(uint8_t*)in,in_len,1);
    }
    else 
    {
        if(ctx->cipherex_in)
            dst_p = abcdk_openssl_cipherex_update_pack(ctx->cipherex_in,(uint8_t*)in,in_len,0);
    }
#endif //OPENSSL_VERSION_NUMBER

    abcdk_rwlock_unlock(ctx->cipher_locker);

    return dst_p;
}

static void _abcdk_sudp_process_input(abcdk_sudp_t *ctx)
{
    abcdk_object_t *enc_p = NULL;
    abcdk_object_t *dec_p = NULL;
    abcdk_sockaddr_t remote_addr = {0};
    char addr_str[100] = {0};
    socklen_t addr_len = 64;
    ssize_t rlen = 0;
    int chk;

NEXT_MSG:
    
    abcdk_object_unref(&enc_p);
    abcdk_object_unref(&dec_p);

    if(!abcdk_atomic_compare(&ctx->worker_flag,0))
        return;

    chk = abcdk_poll(ctx->fd,0x01,1000);
    if(chk <= 0)
        goto NEXT_MSG;

    enc_p = abcdk_object_alloc2(65536);
    if(!enc_p)
        goto NEXT_MSG;

    rlen = recvfrom(ctx->fd,enc_p->pptrs[0],enc_p->sizes[0],0,(struct sockaddr*)&remote_addr,&addr_len);
    if(rlen <= 0)
        goto NEXT_MSG;

    /*fix length.*/
    enc_p->sizes[0] = rlen;

    if(ctx->cfg.ssl_scheme == ABCDK_SUDP_SSL_SCHEME_SKE)
    {
        dec_p = _abcdk_sudp_cipher_update_pack(ctx,enc_p->pptrs[0],enc_p->sizes[0],0);
        if(!dec_p)
        {
            abcdk_sockaddr_to_string(addr_str,&remote_addr,0);
            abcdk_trace_output(LOG_WARNING, "来自(%s)的数据解密失败，丢弃此数据包。\n",addr_str);
            goto NEXT_MSG;
        }
    }
    else
    {
        dec_p = abcdk_object_refer(enc_p);
    }

    if(ctx->cfg.input_cb)
        ctx->cfg.input_cb(ctx->cfg.opaque,&remote_addr,dec_p->pptrs[0], dec_p->sizes[0]);

    goto NEXT_MSG;
}

void _abcdk_sudp_process_input_cb(void *opaque,uint64_t event,void *item)
{
    abcdk_sudp_t *ctx = (abcdk_sudp_t *)opaque;

    /*设置线程名字，日志记录会用到。*/
    abcdk_thread_setname(0, "%x", ctx->index);

    _abcdk_sudp_process_input(ctx);
}

int abcdk_sudp_post(abcdk_sudp_t *ctx,abcdk_sockaddr_t *remote, const void *data,size_t size)
{
    abcdk_object_t *enc_p = NULL;
    int chk;

    assert(ctx != NULL && remote != NULL && data != NULL && size >0 && size <= 64512);

    if(ctx->cfg.ssl_scheme == ABCDK_SUDP_SSL_SCHEME_SKE)
    {
        enc_p = _abcdk_sudp_cipher_update_pack(ctx,(void*)data,size,1);
        if(!enc_p)
            return -1;
        
        chk = sendto(ctx->fd,(void*)enc_p->pptrs[0],enc_p->sizes[0],0,(struct sockaddr*)remote,64);
        abcdk_object_unref(&enc_p);
    }
    else
    {
       chk = sendto(ctx->fd,data,size,0,(struct sockaddr*)remote,64);
    }

    if(chk <= 0)
    {
        abcdk_trace_output(LOG_WARNING, "输出缓慢，当前数据包未能发送。\n");
        return  -1;
    }

    return 0;
}