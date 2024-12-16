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
    /**魔法数。*/
    uint32_t magic;
#define ABCDK_SUDP_MAGIC 123456789

    /**引用计数器。*/
    volatile int refcount;

    /**ASIOEX环境。*/
    abcdk_asioex_t *asioex_ctx;

    /**线程池配置。*/
    abcdk_worker_config_t worker_cfg;

    /**线程池环境。*/
    abcdk_worker_t *worker_ctx;
    
    /*NONCE环境。*/
    abcdk_nonce_t *nonce_ctx;

    /*NONCE前缀。*/
    uint8_t nonce_prefix[16];

}; // abcdk_sudp_t;

/**UDP节点。 */
struct _abcdk_sudp_node
{
    /**魔法数。*/
    uint32_t magic;
#define ABCDK_SUDP_NODE_MAGIC 123456789

    /**引用计数器。*/
    volatile int refcount;

    /**通讯环境指针。*/
    abcdk_sudp_t *ctx;

    /**配置。*/
    abcdk_sudp_config_t cfg;

    /**索引。*/
    uint64_t index;

    /**状态。*/
    volatile int status;
#define ABCDK_SUDP_STATUS_STABLE 1

    /**ASIO环境。*/
    abcdk_asio_t *asio_ctx;

    /**伪句柄。*/
    int64_t pfd;

    /**句柄。*/
    int fd;

    /**密钥同步锁。*/
    abcdk_rwlock_t *cipher_locker;

#ifdef OPENSSL_VERSION_NUMBER
    /**密钥环境。*/
    abcdk_openssl_cipherex_t *cipherex_out;
    abcdk_openssl_cipherex_t *cipherex_in;
#endif // OPENSSL_VERSION_NUMBER

    /**用户环境指针。*/
    abcdk_object_t *userdata;

    /**用户环境销毁函数。*/
    void (*userdata_free_cb)(void *userdata);

    /**接收缓存。*/
    abcdk_object_t *in_buffer;
}; // abcdk_sudp_node_t

static void _abcdk_sudp_ctx_unref(abcdk_sudp_t **ctx)
{
    abcdk_sudp_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->magic == ABCDK_SUDP_MAGIC);

    if (abcdk_atomic_fetch_and_add(&ctx_p->refcount, -1) != 1)
        return;

    assert(ctx_p->refcount == 0);
    ctx_p->magic = 0xcccccccc;

    ABCDK_ASSERT(ctx_p->worker_ctx == NULL, "销毁前必须先停止。");

    abcdk_asioex_destroy(&ctx_p->asioex_ctx);
    abcdk_nonce_destroy(&ctx_p->nonce_ctx);
    abcdk_heap_free(ctx_p);
}

static abcdk_sudp_t *_abcdk_sudp_ctx_refer(abcdk_sudp_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

static void _abcdk_sudp_worker(void *opaque, uint64_t event, void *item);

static abcdk_sudp_t *_abcdk_sudp_ctx_alloc(int worker, int diff)
{
    abcdk_sudp_t *ctx = NULL;

    ctx = (abcdk_sudp_t *)abcdk_heap_alloc(sizeof(abcdk_sudp_t));
    if (!ctx)
        return NULL;

    ctx->magic = ABCDK_SUDP_MAGIC;
    ctx->refcount = 1;

    worker = ABCDK_CLAMP(worker, 1, worker);
    diff = ABCDK_CLAMP(diff, 0, diff);

    ctx->asioex_ctx = abcdk_asioex_create(worker, 99999);
    if (!ctx->asioex_ctx)
        goto ERR;

    ctx->nonce_ctx = abcdk_nonce_create(diff * 1000);
    if (!ctx->nonce_ctx)
        goto ERR;
    
#ifdef OPENSSL_VERSION_NUMBER
    RAND_bytes(ctx->nonce_prefix, 16);
#else //OPENSSL_VERSION_NUMBER
    abcdk_rand_bytes(ctx->nonce_prefix, 16, 5);
#endif //OPENSSL_VERSION_NUMBER

    ctx->worker_cfg.numbers = worker;
    ctx->worker_cfg.opaque = ctx;
    ctx->worker_cfg.process_cb = _abcdk_sudp_worker;
    ctx->worker_ctx = abcdk_worker_start(&ctx->worker_cfg);
    if (!ctx->worker_ctx)
        goto ERR;

    /*每个ASIO分配一个线程处理。*/
    for (int i = 0; i < worker; i++)
        abcdk_worker_dispatch(ctx->worker_ctx, i, (void *)-1);

    return ctx;

ERR:

    _abcdk_sudp_ctx_unref(&ctx);

    return NULL;
}

void abcdk_sudp_unref(abcdk_sudp_node_t **node)
{
    abcdk_sudp_node_t *node_p = NULL;

    if (!node || !*node)
        return;

    node_p = *node;
    *node = NULL;

    assert(node_p->magic == ABCDK_SUDP_NODE_MAGIC);

    if (abcdk_atomic_fetch_and_add(&node_p->refcount, -1) != 1)
        return;

    assert(node_p->refcount == 0);
    node_p->magic = 0xcccccccc;

    if (node_p->userdata_free_cb)
        node_p->userdata_free_cb(node_p->userdata->pptrs[0]);

#ifdef OPENSSL_VERSION_NUMBER
    abcdk_openssl_cipherex_destroy(&node_p->cipherex_in);
    abcdk_openssl_cipherex_destroy(&node_p->cipherex_out);
#endif // OPENSSL_VERSION_NUMBER

    abcdk_closep(&node_p->fd);
    abcdk_rwlock_destroy(&node_p->cipher_locker);
    abcdk_object_unref(&node_p->userdata);
    abcdk_object_unref(&node_p->in_buffer);
    _abcdk_sudp_ctx_unref(&node_p->ctx);
    abcdk_heap_free(node_p);
}

abcdk_sudp_node_t *abcdk_sudp_refer(abcdk_sudp_node_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_sudp_node_t *abcdk_sudp_alloc(abcdk_sudp_t *ctx, size_t userdata, void (*free_cb)(void *userdata))
{
    abcdk_sudp_node_t *node = NULL;

    assert(ctx != NULL && free_cb != NULL);

    node = (abcdk_sudp_node_t *)abcdk_heap_alloc(sizeof(abcdk_sudp_node_t));
    if (!node)
        return NULL;

    node->magic = ABCDK_SUDP_NODE_MAGIC;
    node->refcount = 1;
    node->ctx = _abcdk_sudp_ctx_refer(ctx);
    node->index = abcdk_sequence_num();
    node->status = 0;
    node->pfd = -1;
    node->fd = -1;
    node->cipher_locker = abcdk_rwlock_create();
#ifdef OPENSSL_VERSION_NUMBER
    node->cipherex_in = NULL;
    node->cipherex_out = NULL;
#endif // OPENSSL_VERSION_NUMBER
    node->userdata = abcdk_object_alloc3(userdata, 1);
    node->userdata_free_cb = free_cb;


    return node;
}

uint64_t abcdk_sudp_get_index(abcdk_sudp_node_t *node)
{
    assert(node != NULL);

    return node->index;
}

void *abcdk_sudp_get_userdata(abcdk_sudp_node_t *node)
{
    assert(node != NULL);

    return node->userdata->pptrs[0];
}

int abcdk_sudp_set_timeout(abcdk_sudp_node_t *node, time_t timeout)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    chk = abcdk_asio_timeout(node->asio_ctx, node->pfd, timeout);

    return chk;
}

int abcdk_sudp_cipher_reset(abcdk_sudp_node_t *node, const uint8_t *key, size_t klen, int flag)
{
    int chk = -1;

    assert(node != NULL && key != NULL && klen > 0);

    abcdk_rwlock_wrlock(node->cipher_locker, 1);

#ifdef OPENSSL_VERSION_NUMBER

    if (flag & 0x01)
    {
        /*关闭旧的，并创建新的。*/
        abcdk_openssl_cipherex_destroy(&node->cipherex_in);

        if (node->cfg.ssl_scheme == ABCDK_SUDP_SSL_SCHEME_AES256GCM)
            node->cipherex_in = abcdk_openssl_cipherex_create(4, ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM, key, klen);
        else if (node->cfg.ssl_scheme == ABCDK_SUDP_SSL_SCHEME_AES256CBC)
            node->cipherex_in = abcdk_openssl_cipherex_create(4, ABCDK_OPENSSL_CIPHER_SCHEME_AES256CBC, key, klen);
    }

    if (flag & 0x02)
    {
        /*关闭旧的，并创建新的。*/
        abcdk_openssl_cipherex_destroy(&node->cipherex_out);

        if (node->cfg.ssl_scheme == ABCDK_SUDP_SSL_SCHEME_AES256GCM)
            node->cipherex_out = abcdk_openssl_cipherex_create(4, ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM, key, klen);
        else if (node->cfg.ssl_scheme == ABCDK_SUDP_SSL_SCHEME_AES256CBC)
            node->cipherex_out = abcdk_openssl_cipherex_create(4, ABCDK_OPENSSL_CIPHER_SCHEME_AES256CBC, key, klen);
    }

    /*必须都成功。*/
    chk = ((node->cipherex_in && node->cipherex_out) ? 0 : -1);

#else  // OPENSSL_VERSION_NUMBER
    abcdk_trace_output(LOG_WARNING, "当前环境未包含加密套件，忽略密钥文件。");
#endif // OPENSSL_VERSION_NUMBER

    abcdk_rwlock_unlock(node->cipher_locker);

    return chk;
}

static abcdk_object_t *_abcdk_sudp_cipher_update_pack(abcdk_sudp_node_t *node, void *in, int in_len, int enc)
{
    abcdk_object_t *dst_p = NULL;

    abcdk_rwlock_rdlock(node->cipher_locker, 1);

#ifdef OPENSSL_VERSION_NUMBER
    if (enc)
    {
        if (node->cipherex_out)
            dst_p = abcdk_openssl_cipherex_update_pack(node->cipherex_out, (uint8_t *)in, in_len, 1);
    }
    else
    {
        if (node->cipherex_in)
            dst_p = abcdk_openssl_cipherex_update_pack(node->cipherex_in, (uint8_t *)in, in_len, 0);
    }
#endif // OPENSSL_VERSION_NUMBER

    abcdk_rwlock_unlock(node->cipher_locker);

    return dst_p;
}

static void _abcdk_sudp_close_cb(abcdk_sudp_node_t *node)
{
    if (node->cfg.close_cb)
        node->cfg.close_cb(node);
}

static void _abcdk_sudp_input(abcdk_sudp_node_t *node);

static void _abcdk_sudp_dispatch(abcdk_sudp_t *ctx, uint32_t event, abcdk_sudp_node_t *node)
{
    abcdk_sudp_node_t *node_p;
    int chk;

    if (event & ABCDK_EPOLL_ERROR)
    {
        /*清除状态。*/
        node->status == 0;

        _abcdk_sudp_close_cb(node);

        /*释放事件计数。*/
        abcdk_asio_unref(node->asio_ctx, node->pfd, ABCDK_EPOLL_ERROR);

        /*解除绑定关系。*/
        abcdk_asio_detch(node->asio_ctx, node->pfd);

        /*释放引用后，指针会被清空，因此这里需要复制一下。*/
        node_p = node;

        /*释放引用(关联时的引用)。*/
        abcdk_sudp_unref(&node_p);
    }

    if (event & ABCDK_EPOLL_OUTPUT)
    {
        /*无论连接状态如何，写权利必须内部释放，不能开放给应用层。*/
        abcdk_asio_mark(node->asio_ctx, node->pfd, 0, ABCDK_EPOLL_OUTPUT);

        /*释放事件计数。*/
        abcdk_asio_unref(node->asio_ctx, node->pfd, ABCDK_EPOLL_OUTPUT);
    }

    if (event & ABCDK_EPOLL_INPUT)
    {
        _abcdk_sudp_input(node);

        /*无论连接状态如何，读权利必须内部释放，不能开放给应用层。*/
        abcdk_asio_mark(node->asio_ctx, node->pfd, 0, ABCDK_EPOLL_INPUT);

        /*释放事件计数。*/
        abcdk_asio_unref(node->asio_ctx, node->pfd, ABCDK_EPOLL_INPUT);
    }
}

static void _abcdk_sudp_perform(abcdk_sudp_t *ctx, int idx)
{
    abcdk_asio_t *asio_ctx;
    abcdk_sudp_node_t *node;
    abcdk_epoll_event_t e;
    int chk;

    asio_ctx = abcdk_asioex_dispatch(ctx->asioex_ctx, idx);

    while (1)
    {
        memset(&e, 0, sizeof(abcdk_epoll_event_t));

        chk = abcdk_asio_wait(asio_ctx, &e);
        if (chk <= 0)
            break;

        node = (abcdk_sudp_node_t *)e.data.ptr;

        /*设置线程名字，日志记录会用到。*/
        abcdk_thread_setname(0, "%x", node->index);

        _abcdk_sudp_dispatch(ctx, e.events, node);
    }
}

static void _abcdk_sudp_worker(void *opaque, uint64_t event, void *item)
{
    abcdk_sudp_t *ctx = (abcdk_sudp_t *)opaque;

    _abcdk_sudp_perform(ctx, event);
}

void abcdk_sudp_destroy(abcdk_sudp_t **ctx)
{
    _abcdk_sudp_ctx_unref(ctx);
}

abcdk_sudp_t *abcdk_sudp_create(int worker, int diff)
{
    return _abcdk_sudp_ctx_alloc(worker, diff);
}

void abcdk_sudp_stop(abcdk_sudp_t *ctx)
{
    if (!ctx)
        return;

    /*通知ASIO取消等待。*/
    if (ctx->asioex_ctx)
        abcdk_asioex_abort(ctx->asioex_ctx);

    /*线程池销毁。*/
    abcdk_worker_stop(&ctx->worker_ctx);
}

int abcdk_sudp_enroll(abcdk_sudp_node_t *node, abcdk_sudp_config_t *cfg)
{
    abcdk_sudp_node_t *node_p = NULL;
    epoll_data_t ep_data;
    int sock_flag;
    int chk;

    assert(node != NULL && cfg != NULL);
    assert(cfg->bind_addr.family == AF_INET || cfg->bind_addr.family == AF_INET6);
    ABCDK_ASSERT(cfg->input_cb != NULL, "未绑定通知回调函数，通讯对象无法正常工作。");

    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_sudp_refer(node);

    node_p->cfg = *cfg;

    node_p->status = ABCDK_SUDP_STATUS_STABLE;

    node_p->fd = abcdk_socket(node_p->cfg.bind_addr.family, 1);
    if (node->fd < 0)
        goto ERR;

    /*端口复用，用于快速重启恢复。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(node_p->fd, SOL_SOCKET, SO_REUSEPORT, &sock_flag, 2);
    if (chk != 0)
        goto ERR;

    /*地址复用，用于快速重启恢复。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(node_p->fd, SOL_SOCKET, SO_REUSEADDR, &sock_flag, 2);
    if (chk != 0)
        goto ERR;

    if (node_p->cfg.bind_addr.family == AF_INET6)
    {
        /*IPv6仅支持IPv6。*/
        sock_flag = 1;
        chk = abcdk_sockopt_option_int(node_p->fd, IPPROTO_IPV6, IPV6_V6ONLY, &sock_flag, 2);
        if (chk != 0)
            goto ERR;
    }

    chk = abcdk_bind(node_p->fd, &node_p->cfg.bind_addr);
    if (chk != 0)
        goto ERR;

    if (node_p->cfg.mreq_enable)
    {
        chk = abcdk_socket_option_multicast(node_p->fd, node_p->cfg.bind_addr.family, &node_p->cfg.mreq_addr, 1);
        if (chk != 0)
            goto ERR;
    }

    chk = abcdk_fflag_add(node_p->fd, O_NONBLOCK);
    if (chk != 0)
        goto ERR;

    if (node_p->cfg.bind_ifname && *node_p->cfg.bind_ifname)
    {
        if (getuid() == 0)
        {
            chk = abcdk_socket_option_bindtodevice(node_p->fd, node_p->cfg.bind_ifname);
            if (chk != 0)
                goto ERR;
        }
        else
        {
            abcdk_trace_output(LOG_WARNING, "绑定设备需要root权限支持，忽略配置。");
        }
    }

    node_p->asio_ctx = abcdk_asioex_dispatch(node_p->ctx->asioex_ctx, -1);
    if (!node_p->asio_ctx)
        goto ERR;

    /*节点加入epoll池中。在解除绑定关系前，节点不会被释放。*/
    ep_data.ptr = node_p;
    node_p->pfd = abcdk_asio_attach(node_p->asio_ctx, node_p->fd, &ep_data);
    if (node_p->pfd <= 0)
        goto ERR;

    /*关闭超时。*/
    abcdk_asio_timeout(node_p->asio_ctx, node_p->pfd, 0);
    abcdk_asio_mark(node_p->asio_ctx, node_p->pfd, ABCDK_EPOLL_INPUT, 0);

    return 0;

ERR:

    node->status = 0;
    abcdk_sudp_unref(&node_p);

    return -1;
}

static void _abcdk_sudp_input_cb(abcdk_sudp_node_t *node, abcdk_sockaddr_t *remote, const void *data, size_t size)
{
    char remote_str[100] = {0};
    int chk;


    /*
     * |NONCE    |Data    |
     * |---------|--------|
     * |32 bytes |N Bytes |
     */

    if (size < 32)
        return;
        
    chk = abcdk_nonce_check(node->ctx->nonce_ctx, ABCDK_PTR2U8PTR(data, 0));
    if (chk != 0)
    {
        abcdk_sockaddr_to_string(remote_str, remote, 0);
        abcdk_trace_output(LOG_WARNING, "NONCE无效(%d)，丢弃来自(%s)的数据包。\n", chk, remote_str);
        return;
    }

    if (node->cfg.input_cb)
        node->cfg.input_cb(node, remote, ABCDK_PTR2VPTR(data, 32), size - 32);
}

static void _abcdk_sudp_input_hook(abcdk_sudp_node_t *node, abcdk_sockaddr_t *remote, const void *data, size_t size)
{
    abcdk_object_t *dec_p = NULL;
    char remote_str[100] = {0};
    int chk;

    if (node->cfg.ssl_scheme != ABCDK_SUDP_SSL_SCHEME_RAW)
    {
        dec_p = _abcdk_sudp_cipher_update_pack(node, (void*)data, size, 0);
        if (!dec_p)
        {
            abcdk_sockaddr_to_string(remote_str, remote, 0);
            abcdk_trace_output(LOG_WARNING, "解密错误，丢弃来自(%s)的数据包。\n", remote_str);
            
        }
        else
        {
            _abcdk_sudp_input_cb(node, remote, dec_p->pptrs[0], dec_p->sizes[0]);
            abcdk_object_unref(&dec_p);
        }
    }
    else
    {
        _abcdk_sudp_input_cb(node, remote, data, size);
    }
}

static void _abcdk_sudp_input(abcdk_sudp_node_t *node)
{
    abcdk_sockaddr_t remote_addr = {0};
    socklen_t addr_len = 64;
    ssize_t rlen = 0;
    int chk;

    if (!node->in_buffer)
    {
        node->in_buffer = abcdk_object_alloc2(65536);
        if (!node->in_buffer)
            return;
    }

NEXT_MSG:

    /*收。*/
    rlen = recvfrom(node->fd, node->in_buffer->pptrs[0], node->in_buffer->sizes[0], 0, (struct sockaddr *)&remote_addr, &addr_len);
    if (rlen <= 0)
    {
        abcdk_asio_mark(node->asio_ctx, node->pfd, ABCDK_EPOLL_INPUT, 0);
        return;
    }

    _abcdk_sudp_input_hook(node, &remote_addr, node->in_buffer->pptrs[0], rlen);

    /*继续读取缓存内可能存在数据，直到为空。*/
    goto NEXT_MSG;
}

static int _abcdk_sudp_output_hook(abcdk_sudp_node_t *node, abcdk_sockaddr_t *remote, const void *data, size_t size)
{
    abcdk_object_t *enc_p = NULL;
    int chk;

    if (node->cfg.ssl_scheme != ABCDK_SUDP_SSL_SCHEME_RAW)
    {
        enc_p = _abcdk_sudp_cipher_update_pack(node, (void *)data, size, 1);
        if (!enc_p)
            return -1;

        chk = sendto(node->fd, (void *)enc_p->pptrs[0], enc_p->sizes[0], 0, (struct sockaddr *)remote, 64);
        abcdk_object_unref(&enc_p);
    }
    else
    {
        chk = sendto(node->fd, data, size, 0, (struct sockaddr *)remote, 64);
    }

    if (chk <= 0)
    {
        abcdk_trace_output(LOG_DEBUG, "输出缓慢，当前数据包未能发送。\n");
        return -1;
    }

    return 0;
}

int abcdk_sudp_post(abcdk_sudp_node_t *node, abcdk_sockaddr_t *remote, const void *data, size_t size)
{
    abcdk_bit_t reqbit = {0};
    abcdk_object_t *reqbuf = NULL;
    uint8_t nonce[32] = {0};
    int chk;

    assert(node != NULL && remote != NULL && data != NULL && size > 0 && size <= 64000);

    if (node->status == 0)
        return -1;

    reqbuf = abcdk_object_alloc2(32 + size);
    if (!reqbuf)
        return -1;

    reqbit.data = reqbuf->pptrs[0];
    reqbit.size = reqbuf->sizes[0];

    abcdk_nonce_generate(node->ctx->nonce_ctx, node->ctx->nonce_prefix, nonce);

    /*
     * |NONCE    |Data    |
     * |---------|--------|
     * |32 bytes |N Bytes |
     */

    abcdk_bit_write_buffer(&reqbit, nonce, 32);
    abcdk_bit_write_buffer(&reqbit, data, size);

    chk = _abcdk_sudp_output_hook(node, remote, reqbuf->pptrs[0], reqbuf->sizes[0]);
    abcdk_object_unref(&reqbuf);

    if (chk != 0)
        return -1;

    return 0;
}