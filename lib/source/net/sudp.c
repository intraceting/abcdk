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

    /**句柄。*/
    int fd;

    /**发送算法。 */
    abcdk_wred_t *out_wred;

    /**发送队列长度。 */
    int out_len;

    /**发送队列。*/
    abcdk_tree_t *out_queue;

    /**发送队列锁。*/
    abcdk_mutex_t *out_locker;

    /**密钥状态。0 禁用, !0 启用。*/
    volatile int cipher_ok;

    /**密钥同步锁。*/
    abcdk_rwlock_t *cipher_locker;

#ifdef OPENSSL_VERSION_NUMBER
    /**密钥环境。*/
    abcdk_openssl_cipher_t *cipher_out;
    abcdk_openssl_cipher_t *cipher_in;
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

    abcdk_wred_destroy(&ctx_p->out_wred);
    abcdk_tree_free(&ctx_p->out_queue);
    abcdk_mutex_destroy(&ctx_p->out_locker);
    abcdk_rwlock_destroy(&ctx_p->cipher_locker);
#ifdef OPENSSL_VERSION_NUMBER
    abcdk_openssl_cipher_destroy(&ctx_p->cipher_in);
    abcdk_openssl_cipher_destroy(&ctx_p->cipher_out);
#endif //OPENSSL_VERSION_NUMBER
    abcdk_closep(&ctx_p->fd);
    
    abcdk_heap_free(ctx_p);
}

static void _abcdk_sudp_process_cb(void *opaque,uint64_t event,void *item);

static void _abcdk_sudp_fix_cfg(abcdk_sudp_t *ctx)
{
    if(ctx->cfg.out_min_th <= 0)
        ctx->cfg.out_min_th = 200;
    else 
        ctx->cfg.out_min_th = ABCDK_CLAMP(ctx->cfg.out_min_th,200,600);

    if(ctx->cfg.out_max_th <= 0)
        ctx->cfg.out_max_th = 400;
    else 
        ctx->cfg.out_max_th = ABCDK_CLAMP(ctx->cfg.out_max_th,400,800);

    if(ctx->cfg.out_weight <= 0)
        ctx->cfg.out_weight = 2;
    else 
        ctx->cfg.out_weight = ABCDK_CLAMP(ctx->cfg.out_weight,1,99);
    
    if(ctx->cfg.out_prob <= 0)
        ctx->cfg.out_prob = 2;
    else 
        ctx->cfg.out_prob = ABCDK_CLAMP(ctx->cfg.out_prob,1,99);

    /*最小阈值和最大阈值必须符合区间要求。*/
    if (ctx->cfg.out_min_th > ctx->cfg.out_max_th)
        ABCDK_INTEGER_SWAP(ctx->cfg.out_min_th, ctx->cfg.out_max_th);
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

    ctx->fd = -1;
    ctx->out_len = 0;

    ctx->out_queue = abcdk_tree_alloc3(1);
    if(!ctx->out_queue)
        goto ERR;

    ctx->out_locker = abcdk_mutex_create();
    if(!ctx->out_locker)
        goto ERR;

    ctx->cipher_locker = abcdk_rwlock_create();
    if(!ctx->out_locker)
        goto ERR;

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

    ctx->out_wred = abcdk_wred_create(ctx->cfg.out_min_th, ctx->cfg.out_max_th,
                                      ctx->cfg.out_weight, ctx->cfg.out_prob);
    if (!ctx->out_wred)
        goto ERR;

    ctx->worker_cfg.numbers = 2;
    ctx->worker_cfg.opaque = ctx;
    ctx->worker_cfg.process_cb = _abcdk_sudp_process_cb;

    ctx->worker_ctx = abcdk_worker_start(&ctx->worker_cfg);
    if(!ctx->worker_ctx)
        goto ERR;

    abcdk_worker_dispatch(ctx->worker_ctx, 1, (void *)-1);
    abcdk_worker_dispatch(ctx->worker_ctx, 2, (void *)-1);

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
        /*关闭旧的。*/
        abcdk_openssl_cipher_destroy(&ctx->cipher_in);
        
        /*创建新的。*/
        ctx->cipher_in = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM,key,klen);
    }

    if(flag & 0x02)
    {
        /*关闭旧的。*/
        abcdk_openssl_cipher_destroy(&ctx->cipher_out);
        
        /*创建新的。*/
        ctx->cipher_out = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM,key,klen);
    }

    /*必须都成功。*/
    chk = ((ctx->cipher_in && ctx->cipher_out) ? 0 : -1);

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
        if(ctx->cipher_out)
            dst_p = abcdk_openssl_cipher_update_pack(ctx->cipher_out,(uint8_t*)in,in_len,1);
    }
    else 
    {
        if(ctx->cipher_in)
            dst_p = abcdk_openssl_cipher_update_pack(ctx->cipher_in,(uint8_t*)in,in_len,0);
    }
#endif //OPENSSL_VERSION_NUMBER

    abcdk_rwlock_unlock(ctx->cipher_locker);

    return dst_p;
}

static void _abcdk_sudp_process_input(abcdk_sudp_t *ctx)
{
    abcdk_object_t *enc_p = NULL;
    abcdk_object_t *dec_p = NULL;
    char addrbuf[100] = {0};
    abcdk_sockaddr_t remote;
    socklen_t addr_len = 64;
    ssize_t rlen = 0;
    int chk;

NEXT_MSG:
    
    abcdk_object_unref(&enc_p);
    abcdk_object_unref(&dec_p);
    memset(addrbuf,0,100);

    if(!abcdk_atomic_compare(&ctx->worker_flag,0))
        return;

    chk = abcdk_poll(ctx->fd,0x01,1000);
    if(chk <= 0)
        goto NEXT_MSG;

    enc_p = abcdk_object_alloc2(65536);
    if(!enc_p)
        goto NEXT_MSG;

    rlen = recvfrom(ctx->fd,enc_p->pptrs[0],enc_p->sizes[0],0,&remote.addr,&addr_len);
    if(rlen <= 0)
        goto NEXT_MSG;

    /*fix length.*/
    enc_p->sizes[0] = rlen;

    abcdk_sockaddr_to_string(addrbuf,&remote,0);
    
    if(ctx->cfg.ssl_scheme == ABCDK_SUDP_SSL_SCHEME_SKE)
    {
        dec_p = _abcdk_sudp_cipher_update_pack(ctx,enc_p->pptrs[0],enc_p->sizes[0],0);
        if(!dec_p)
        {
            abcdk_trace_output(LOG_WARNING, "来自(%s)的数据解密失败，丢弃此数据包。\n",addrbuf);

            goto NEXT_MSG;
        }
    }
    else 
    {
        dec_p = abcdk_object_refer(enc_p);
    }

    if(ctx->cfg.input_cb)
        ctx->cfg.input_cb(ctx->cfg.opaque,&remote,dec_p->pptrs[0], dec_p->sizes[0]);

    goto NEXT_MSG;
}

static void _abcdk_sudp_process_output(abcdk_sudp_t *ctx)
{
    abcdk_tree_t *p = NULL;
    abcdk_object_t *enc_p = NULL;
    ssize_t slen = 0;
    int chk;

NEXT_MSG:

    abcdk_object_unref(&enc_p);

    if(!abcdk_atomic_compare(&ctx->worker_flag,0))
        return;

    chk = abcdk_poll(ctx->fd,0x02,1000);
    if(chk <= 0)
        goto NEXT_MSG;

    /*从队列头部开始发送。*/
    abcdk_mutex_lock(ctx->out_locker, 1);
    p = abcdk_tree_child(ctx->out_queue, 1);
    if (!p)
        abcdk_mutex_wait(ctx->out_locker, 1000);
    abcdk_mutex_unlock(ctx->out_locker);

    if(!p)
        goto NEXT_MSG;

    if(ctx->cfg.ssl_scheme == ABCDK_SUDP_SSL_SCHEME_SKE)
    {
        enc_p = _abcdk_sudp_cipher_update_pack(ctx,p->obj->pptrs[0],p->obj->sizes[0],1);
        if(!enc_p)
            goto NEXT_MSG;
    }
    else
    {
        enc_p = abcdk_object_refer(p->obj);
    }

    slen = sendto(ctx->fd,(void*)enc_p->pptrs[0],enc_p->sizes[0],0,(struct sockaddr*)p->obj->pptrs[1],p->obj->sizes[1]);
    if(slen <= 0 && errno != EINVAL)
        goto NEXT_MSG;

    /*从队列中移除节点。*/
    abcdk_mutex_lock(ctx->out_locker, 1);
    abcdk_tree_unlink(p);
    ctx->out_len -= 1;
    abcdk_mutex_unlock(ctx->out_locker);

    /*删除节点。*/
    abcdk_tree_free(&p);

    goto NEXT_MSG;
}

static void _abcdk_sudp_perform(abcdk_sudp_t *ctx, uint64_t event)
{
    if (event == 1)
        _abcdk_sudp_process_input(ctx);
    else if (event == 2)
        _abcdk_sudp_process_output(ctx);
}

void _abcdk_sudp_process_cb(void *opaque,uint64_t event,void *item)
{
    abcdk_sudp_t *ctx = (abcdk_sudp_t *)opaque;

    _abcdk_sudp_perform(ctx,event);
}

int abcdk_sudp_post(abcdk_sudp_t *ctx,abcdk_object_t *data)
{
    abcdk_tree_t *p;
    int chk;

    assert(ctx != NULL && data != NULL);
    assert(data->pptrs[0] != NULL && data->sizes[0] > 0 && data->sizes[0] <= 64512);
    assert(data->pptrs[1] != NULL && data->sizes[1] > 0 && data->sizes[1] <= 64);

    p = abcdk_tree_alloc(data);
    if(!p)
        return -1;

    /*添加到队列末尾。*/
    
    abcdk_mutex_lock(ctx->out_locker,1);

    /*根据WRED算法决定是否添加到队列中。*/
    chk = abcdk_wred_update(ctx->out_wred, ctx->out_len + 1);
    if (chk == 0)
    {
        abcdk_tree_insert2(ctx->out_queue, p, 0);
        ctx->out_len += 1;
        abcdk_mutex_signal(ctx->out_locker,0);
    }
    else
    {
        abcdk_trace_output(LOG_WARNING, "输出缓慢，队列积压过长(len=%d)，丢弃当前数据包(size=%zd)。\n",ctx->out_len, p->obj->sizes[0]);

        abcdk_tree_free(&p);
    }
    
    abcdk_mutex_unlock(ctx->out_locker);

    return 0;
}

int abcdk_sudp_post_buffer(abcdk_sudp_t *ctx,abcdk_sockaddr_t *remote, const void *data,size_t size)
{
    abcdk_object_t *src_p = NULL;
    int chk;

    assert(ctx != NULL && remote != NULL && data != NULL && size >0 && size <= 64512);

    size_t sizes[] = {size,64};
    src_p = abcdk_object_alloc(sizes,2,0);
    if(!src_p)
        return -1;

    memcpy(src_p->pptrs[0],data,size);
    memcpy(src_p->pptrs[1],remote,64);

    chk = abcdk_sudp_post(ctx,src_p);
    if(chk == 0)
        return 0;

    abcdk_object_unref(&src_p);
    return -1;
}