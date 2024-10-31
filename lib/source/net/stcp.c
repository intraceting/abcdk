/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/net/stcp.h"

/**简单的异步TCP通讯。 */
struct _abcdk_stcp
{
    /**魔法数。*/
    uint32_t magic;
#define ABCDK_STCP_MAGIC 123456789

    /**引用计数器。*/
    volatile int refcount;

    /**ASIOEX环境。*/
    abcdk_asioex_t *asioex_ctx;

    /**线程池配置。*/
    abcdk_worker_config_t worker_cfg;

    /**线程池环境。*/
    abcdk_worker_t *worker_ctx;

}; // abcdk_stcp_t;

/** 异步TCP节点。 */
struct _abcdk_stcp_node
{
    /**魔法数。*/
    uint32_t magic;
#define ABCDK_STCP_NODE_MAGIC 123456789

    /**引用计数器。*/
    volatile int refcount;

    /**
     * 通讯环境指针。
     *
     * @note 仅复制。
     */
    abcdk_stcp_t *ctx;

    /**配置。*/
    abcdk_stcp_config_t cfg;

    /**
     * 索引。
     */
    uint64_t index;

    /**标识句柄来源。*/
    volatile int flag;
#define ABCDK_STCP_FLAG_CLIENT 1
#define ABCDK_STCP_FLAG_LISTEN 2
#define ABCDK_STCP_FLAG_ACCPET 3

    /**标识当前句柄状态。*/
    volatile int status;
#define ABCDK_STCP_STATUS_STABLE 1
#define ABCDK_STCP_STATUS_SYNC 2
#define ABCDK_STCP_STATUS_SYNC_PKI 3

    /**本机地址。*/
    abcdk_sockaddr_t local;

    /**远端地址。*/
    abcdk_sockaddr_t remote;

    /**ASIO环境。*/
    abcdk_asio_t *asio_ctx;

    /**伪句柄。*/
    int64_t pfd;

    /**句柄。*/
    int fd;

    /**OpenSSL环境指针。*/
    SSL_CTX *openssl_ctx;

    /**OpenSSL环境指针。*/
    SSL *openssl_ssl;

    /**BIO环境指针。*/
    BIO *openssl_bio;

    /**MaskSSL环境指针。*/
    abcdk_maskssl_t *maskssl_ssl;

    /**读线程。*/
    volatile pthread_t recv_leader;

    /**写线程。*/
    volatile pthread_t send_leader;

    /**用户环境指针。*/
    abcdk_object_t *userdata;

    /**用户环境销毁函数。*/
    void (*userdata_free_cb)(void *userdata);

    /**发送算法。 */
    abcdk_wred_t *out_wred;

    /**发送队列长度。 */
    int out_len;

    /**发送队列。*/
    abcdk_tree_t *out_queue;

    /**发送队列锁。*/
    abcdk_spinlock_t *out_locker;

    /**发送游标。*/
    size_t out_pos;

    /**接收缓存。*/
    abcdk_object_t *in_buffer;

    /**来自哪个监听节点。*/
    abcdk_stcp_node_t *from_listen;

}; // abcdk_stcp_node_t;

static void _abcdk_stcp_ctx_unref(abcdk_stcp_t **ctx)
{
    abcdk_stcp_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->magic == ABCDK_STCP_MAGIC);

    if (abcdk_atomic_fetch_and_add(&ctx_p->refcount, -1) != 1)
        return;

    assert(ctx_p->refcount == 0);
    ctx_p->magic = 0xcccccccc;

    /*如果创建成功，先通知取消等待，否则工作线程无法终止。*/
    if (ctx_p->asioex_ctx)
        abcdk_asioex_abort(ctx_p->asioex_ctx);

    abcdk_worker_stop(&ctx_p->worker_ctx);
    abcdk_asioex_destroy(&ctx_p->asioex_ctx);
    abcdk_heap_free(ctx_p);
}

static abcdk_stcp_t *_abcdk_stcp_ctx_refer(abcdk_stcp_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

static void _abcdk_stcp_worker(void *opaque, uint64_t event, void *item);

static abcdk_stcp_t *_abcdk_stcp_ctx_alloc(int worker)
{
    abcdk_stcp_t *ctx = NULL;

    ctx = (abcdk_stcp_t *)abcdk_heap_alloc(sizeof(abcdk_stcp_t));
    if (!ctx)
        return NULL;

    ctx->magic = ABCDK_STCP_MAGIC;
    ctx->refcount = 1;

    worker = ABCDK_CLAMP(worker, 1, worker);

    ctx->asioex_ctx = abcdk_asioex_create(worker, 99999);
    if (!ctx->asioex_ctx)
        goto ERR;

    ctx->worker_cfg.numbers = worker;
    ctx->worker_cfg.opaque = ctx;
    ctx->worker_cfg.process_cb = _abcdk_stcp_worker;
    ctx->worker_ctx = abcdk_worker_start(&ctx->worker_cfg);
    if (!ctx->worker_ctx)
        goto ERR;

    /*每个ASIO分配一个线程处理。*/
    for (int i = 0; i < worker; i++)
        abcdk_worker_dispatch(ctx->worker_ctx, i, (void *)-1);

    return ctx;

ERR:

    _abcdk_stcp_ctx_unref(&ctx);

    return NULL;
}

void abcdk_stcp_unref(abcdk_stcp_node_t **node)
{
    abcdk_stcp_node_t *node_p = NULL;

    if (!node || !*node)
        return;

    node_p = *node;
    *node = NULL;

    assert(node_p->magic == ABCDK_STCP_NODE_MAGIC);

    if (abcdk_atomic_fetch_and_add(&node_p->refcount, -1) != 1)
        return;

    assert(node_p->refcount == 0);
    node_p->magic = 0xcccccccc;

    if (node_p->userdata_free_cb)
        node_p->userdata_free_cb(node_p->userdata->pptrs[0]);

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_free(&node_p->openssl_ssl);
    abcdk_openssl_BIO_destroy(&node_p->openssl_bio);
    abcdk_openssl_ssl_ctx_free(&node_p->openssl_ctx);
#endif // HEADER_SSL_H

    abcdk_maskssl_destroy(&node_p->maskssl_ssl);

    /*直接关闭，快速回收资源，不会处于time_wait状态。*/
    if (node_p->fd >= 0)
        abcdk_socket_option_linger_set(node_p->fd, 1, 0);

    abcdk_closep(&node_p->fd);
    abcdk_object_unref(&node_p->userdata);
    abcdk_wred_destroy(&node_p->out_wred);
    abcdk_tree_free(&node_p->out_queue);
    abcdk_spinlock_destroy(&node_p->out_locker);
    abcdk_object_unref(&node_p->in_buffer);
    abcdk_stcp_unref(&node_p->from_listen);
    _abcdk_stcp_ctx_unref(&node_p->ctx);
    abcdk_heap_free(node_p);
}

abcdk_stcp_node_t *abcdk_stcp_refer(abcdk_stcp_node_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_stcp_node_t *abcdk_stcp_alloc(abcdk_stcp_t *ctx, size_t userdata, void (*free_cb)(void *userdata))
{
    abcdk_stcp_node_t *node = NULL;

    assert(ctx != NULL && free_cb != NULL);

    node = (abcdk_stcp_node_t *)abcdk_heap_alloc(sizeof(abcdk_stcp_node_t));
    if (!node)
        return NULL;

    node->magic = ABCDK_STCP_NODE_MAGIC;
    node->refcount = 1;
    node->ctx = _abcdk_stcp_ctx_refer(ctx);
    node->index = abcdk_sequence_num();
    node->pfd = -1;
    node->fd = -1;
    node->userdata = abcdk_object_alloc3(userdata, 1);
    node->userdata_free_cb = free_cb;
    node->out_wred = NULL;
    node->out_queue = abcdk_tree_alloc3(1);
    node->out_locker = abcdk_spinlock_create();
    node->out_pos = 0;
    node->in_buffer = NULL;
    node->from_listen = NULL;
    node->openssl_ctx = NULL;
    node->openssl_ssl = NULL;
    node->openssl_bio = NULL;
    node->maskssl_ssl = NULL;

    return node;
}

void abcdk_stcp_trace_output(abcdk_stcp_node_t *node, int type, const char *fmt, ...)
{
    char new_tname[18] = {0}, old_tname[18] = {0};

    assert(node != NULL);

    snprintf(new_tname, 16, "%x", node->index);

#ifdef __USE_GNU
    pthread_getname_np(pthread_self(), old_tname, 18);
    pthread_setname_np(pthread_self(), new_tname);
#endif //__USE_GNU

    va_list vp;
    va_start(vp, fmt);
    abcdk_trace_voutput(type, fmt, vp);
    va_end(vp);

#ifdef __USE_GNU
    pthread_setname_np(pthread_self(), old_tname);
#endif //__USE_GNU
}

uint64_t abcdk_stcp_get_index(abcdk_stcp_node_t *node)
{
    assert(node != NULL);

    return node->index;
}

SSL *abcdk_stcp_openssl_get_handle(abcdk_stcp_node_t *node)
{
    assert(node != NULL);

    return node->openssl_ssl;
}

char *abcdk_stcp_openssl_get_alpn_selected(abcdk_stcp_node_t *node, char proto[255 + 1])
{
    int chk;

    assert(node != NULL && proto != NULL);

    if (!node->openssl_ssl)
        return NULL;

#ifdef HEADER_SSL_H

    chk = abcdk_openssl_ssl_get_alpn_selected(node->openssl_ssl, proto);
    if (chk != 0)
        return proto;

#endif // HEADER_SSL_H

    return NULL;
}

void *abcdk_stcp_get_userdata(abcdk_stcp_node_t *node)
{
    assert(node != NULL);

    return node->userdata->pptrs[0];
}

int abcdk_stcp_set_timeout(abcdk_stcp_node_t *node, time_t timeout)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if (!node->flag || !node->status)
        return -3;

    chk = abcdk_asio_timeout(node->asio_ctx, node->pfd, timeout);

    return chk;
}

int abcdk_stcp_get_sockaddr(abcdk_stcp_node_t *node, abcdk_sockaddr_t *local, abcdk_sockaddr_t *remote)
{
    assert(node != NULL);

    if (local && node->local.family)
        abcdk_sockaddr_copy(&node->local, local);

    if (remote && node->remote.family)
        abcdk_sockaddr_copy(&node->remote, remote);

    return 0;
}

int abcdk_stcp_get_sockaddr_str(abcdk_stcp_node_t *node, char local[NAME_MAX], char remote[NAME_MAX])
{
    assert(node != NULL);

    if (local && node->local.family)
        abcdk_sockaddr_to_string(local, &node->local, 1);

    if (remote && node->remote.family)
        abcdk_sockaddr_to_string(remote, &node->remote, 1);

    return 0;
}

ssize_t abcdk_stcp_recv(abcdk_stcp_node_t *node, void *buf, size_t size)
{
    ssize_t rsize = 0, rsize_all = 0;
    int chk;

    assert(node != NULL && buf != NULL && size > 0);

    /*仅工作线程拥有读权利。*/
    chk = abcdk_thread_leader_test(&node->recv_leader);
    ABCDK_ASSERT(chk == 0, "当前线程没有读权利。");

    while (rsize_all < size)
    {
        if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI || node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI_ON_SK)
            rsize = SSL_read(node->openssl_ssl, ABCDK_PTR2PTR(void, buf, rsize_all), size - rsize_all);
        else if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_SK)
            rsize = abcdk_maskssl_read(node->maskssl_ssl, ABCDK_PTR2PTR(void, buf, rsize_all), size - rsize_all);
        else
            rsize = read(node->fd, ABCDK_PTR2PTR(void, buf, rsize_all), size - rsize_all);

        if (rsize <= 0)
            break;

        rsize_all += rsize;
    }

    return rsize_all;
}

int abcdk_stcp_recv_watch(abcdk_stcp_node_t *node)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if (!node->flag || !node->status)
        return -3;

    chk = abcdk_asio_mark(node->asio_ctx, node->pfd, ABCDK_EPOLL_INPUT, 0);

    return chk;
}

ssize_t abcdk_stcp_send(abcdk_stcp_node_t *node, void *buf, size_t size)
{
    ssize_t wsize = 0, wsize_all = 0;
    int chk;

    assert(node != NULL && buf != NULL && size > 0);

    /*仅工作线程拥有写权利。*/
    chk = abcdk_thread_leader_test(&node->send_leader);
    ABCDK_ASSERT(chk == 0, "当前线程没有读权利。");

    while (wsize_all < size)
    {
        if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI || node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI_ON_SK)
            wsize = SSL_write(node->openssl_ssl, ABCDK_PTR2PTR(void, buf, wsize_all), size - wsize_all);
        else if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_SK)
            wsize = abcdk_maskssl_write(node->maskssl_ssl, ABCDK_PTR2PTR(void, buf, wsize_all), size - wsize_all);
        else
            wsize = write(node->fd, ABCDK_PTR2PTR(void, buf, wsize_all), size - wsize_all);

        if (wsize <= 0)
            break;

        wsize_all += wsize;
    }

    return wsize_all;
}

int abcdk_stcp_send_watch(abcdk_stcp_node_t *node)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if (!node->flag || !node->status)
        return -3;

    chk = abcdk_asio_mark(node->asio_ctx, node->pfd, ABCDK_EPOLL_OUTPUT, 0);

    return chk;
}

void _abcdk_stcp_cleanup_cb(epoll_data_t *data, void *opaque)
{
    abcdk_stcp_t *ctx = (abcdk_stcp_t *)opaque;
    abcdk_stcp_node_t *node = NULL;

    node = (abcdk_stcp_node_t *)data->ptr;
    abcdk_stcp_unref(&node);
}

void _abcdk_stcp_prepare_cb(abcdk_stcp_node_t **node, abcdk_stcp_node_t *listen)
{
    /*通知应用层处理事件。*/
    if (listen->cfg.prepare_cb)
        listen->cfg.prepare_cb(node, listen);
}

/*声明输入事件钩子函数。*/
static void _abcdk_stcp_input_hook(abcdk_stcp_node_t *node);

/*声明输出事件钩子函数。*/
static void _abcdk_stcp_output_hook(abcdk_stcp_node_t *node);

void _abcdk_stcp_event_cb(abcdk_stcp_node_t *node, uint32_t event, int *result)
{
    if (event == ABCDK_STCP_EVENT_INPUT)
    {
        abcdk_thread_leader_vote(&node->recv_leader);
        _abcdk_stcp_input_hook(node);
        abcdk_thread_leader_quit(&node->recv_leader);
    }
    else if (event == ABCDK_STCP_EVENT_OUTPUT)
    {
        abcdk_thread_leader_vote(&node->send_leader);
        _abcdk_stcp_output_hook(node);
        abcdk_thread_leader_quit(&node->send_leader);
    }
    else
    {
        node->cfg.event_cb(node, event, result);
    }
}

void _abcdk_stcp_accept(abcdk_stcp_node_t *listen)
{
    abcdk_stcp_node_t *node = NULL;
    epoll_data_t ep_data;
    int chk;

    /*通知初始化。*/
    _abcdk_stcp_prepare_cb(&node, listen);
    if (!node)
        return;

    /*配置参数。*/
    node->flag = ABCDK_STCP_FLAG_ACCPET;
    node->status = ABCDK_STCP_STATUS_SYNC;

    /*记住来源。*/
    node->from_listen = abcdk_stcp_refer(listen);

    /*复制通讯环境指针。*/
    node->ctx = listen->ctx;

    /*复制监听环境的配置。*/
    node->cfg = listen->cfg;

    /*每次取出一个句柄。*/
    node->fd = abcdk_accept(listen->fd, &node->remote);
    if (node->fd < 0)
        goto ERR;

    /*通知应用层新连接到来。*/
    _abcdk_stcp_event_cb(node, ABCDK_STCP_EVENT_ACCEPT, &chk);
    if (chk != 0)
        goto ERR;

    chk = abcdk_fflag_add(node->fd, O_NONBLOCK);
    if (chk != 0)
        goto ERR;

    node->out_wred = abcdk_wred_create(node->cfg.out_hook_min_th, node->cfg.out_hook_max_th,
                                       node->cfg.out_hook_weight, node->cfg.out_hook_prob);
    if (!node->out_wred)
        goto ERR;

    node->asio_ctx = abcdk_asioex_dispatch(node->ctx->asioex_ctx, -1);
    if (!node->asio_ctx)
        goto ERR;

    ep_data.ptr = node;
    node->pfd = abcdk_asio_attach(node->asio_ctx, node->fd, &ep_data);
    if (node->pfd <= 0)
        goto ERR;

    abcdk_asio_timeout(node->asio_ctx, node->pfd, 180);

    /*注册输出事件用于探测连接状态。*/
    abcdk_asio_mark(node->asio_ctx, node->pfd, ABCDK_EPOLL_OUTPUT, 0);

    return;

ERR:

    /*通知关闭。*/
    _abcdk_stcp_event_cb(node, ABCDK_STCP_EVENT_INTERRUPT, &chk);
    abcdk_stcp_unref(&node);

    return;
}

#ifdef HEADER_SSL_H
static int _abcdk_stcp_openssl_verify_result(abcdk_stcp_node_t *node)
{
    char remote_addr[NAME_MAX] = {0};
    int chk;

    abcdk_stcp_get_sockaddr_str(node, NULL, remote_addr);

    X509 *cert = SSL_get_peer_certificate(node->openssl_ssl);
    if (cert)
    {
        abcdk_object_t *info = abcdk_openssl_cert_dump(cert);
        if (info)
        {
            abcdk_stcp_trace_output(node, LOG_INFO, "远端(%s)的证书信息：\n%s", remote_addr, info->pstrs[0]);
            abcdk_object_unref(&info);
        }

        X509_free(cert);
    }

    if (node->cfg.pki_check_cert)
    {
        chk = SSL_get_verify_result(node->openssl_ssl);
        if (chk != X509_V_OK)
        {
            abcdk_stcp_trace_output(node, LOG_INFO, "远端(%s)的证书验证有错误发生(ssl-errno=%d)。", remote_addr, chk);
            return -1;
        }
    }

    return 0;
}

static void _abcdk_stcp_openssl_dump_errmsg(abcdk_stcp_node_t *node, unsigned long e)
{
    char remote_addr[NAME_MAX] = {0};
    char local_addr[NAME_MAX] = {0};
    char errmsg[NAME_MAX] = {0};

    ERR_error_string_n(e, errmsg, NAME_MAX - 1);

    abcdk_stcp_get_sockaddr_str(node, local_addr, remote_addr);

    abcdk_stcp_trace_output(node, LOG_INFO, "本机(%s)与远端(%s)的连接有错误发生(%s)。", local_addr, remote_addr, errmsg);
}

#endif // HEADER_SSL_H

static void _abcdk_stcp_handshake_sync_after(abcdk_stcp_node_t *node)
{
    socklen_t sock_len = 0;
    int sock_flag = 1;
    struct timeval tv;
    int chk;

    /*获取远程地址。*/
    if (!node->remote.family)
    {
        sock_len = sizeof(abcdk_sockaddr_t);
        getpeername(node->fd, &node->remote.addr, &sock_len);
    }

    /*获取本机地址。*/
    if (!node->local.family)
    {
        sock_len = sizeof(abcdk_sockaddr_t);
        getsockname(node->fd, &node->local.addr, &sock_len);
    }

    /*去掉默认的发和收超时设置。*/
    tv.tv_sec = tv.tv_usec = 0;
    chk = abcdk_sockopt_option_timeout(node->fd, SO_RCVTIMEO, &tv, 2);
    chk = abcdk_sockopt_option_timeout(node->fd, SO_SNDTIMEO, &tv, 2);

    // /*设置发送缓存区。*/
    // sock_flag = 512*1024;
    // chk = abcdk_sockopt_option_int(node->fd, SOL_SOCKET, SO_SNDBUF, &sock_flag, 2);

    // /*设置接收缓存区。*/
    // sock_flag = 512*1024;
    // chk = abcdk_sockopt_option_int(node->fd, SOL_SOCKET, SO_RCVBUF, &sock_flag, 2);

    /*修改保活参数，以防在远程断电的情况下本地无法检测到连接断开信号。*/

    /*开启keepalive属性*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(node->fd, SOL_SOCKET, SO_KEEPALIVE, &sock_flag, 2);

    /*连接在60秒内没有任何数据往来，则进行探测。*/
    sock_flag = 60;
    chk = abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_KEEPIDLE, &sock_flag, 2);

    /*探测时发包的时间间隔为5秒。*/
    sock_flag = 5;
    chk = abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_KEEPINTVL, &sock_flag, 2);

    /*探测尝试的次数.如果第一次探测包就收到响应，则后两次的不再发。*/
    sock_flag = 3;
    chk = abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_KEEPCNT, &sock_flag, 2);

    /*关闭延迟发送。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_NODELAY, &sock_flag, 2);
}

static int _abcdk_stcp_handshake_ssl_init(abcdk_stcp_node_t *node)
{
    if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_RAW)
    {
        return 0;
    }
    else if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI)
    {
#ifdef HEADER_SSL_H
        if (!node->openssl_ssl)
            node->openssl_ssl = abcdk_openssl_ssl_alloc(node->flag == ABCDK_STCP_FLAG_ACCPET ? node->from_listen->openssl_ctx : node->openssl_ctx);
        else
            return -16;

        if (!node->openssl_ssl)
        {
            abcdk_stcp_trace_output(node, LOG_WARNING, "内存或资源不足，无法创建OpenSSL环境(scheme=%d)。", node->cfg.ssl_scheme);
            return -1;
        }

        SSL_set_fd(node->openssl_ssl, node->fd);

        if (node->flag == ABCDK_STCP_FLAG_ACCPET)
            SSL_set_accept_state(node->openssl_ssl);
        else if (node->flag == ABCDK_STCP_FLAG_CLIENT)
            SSL_set_connect_state(node->openssl_ssl);
        else
            return -22;
#else
        abcdk_stcp_trace_output(node, LOG_WARNING, "构建时未包含相关组件，无法创建OpenSSL环境。");
        return -22;
#endif // HEADER_SSL_H
    }
    else if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_SK)
    {
        if (!node->maskssl_ssl)
            node->maskssl_ssl = abcdk_maskssl_create_from_file(node->cfg.sk_key_cipher, node->cfg.sk_key_file);
        else
            return -16;

        if (!node->maskssl_ssl)
        {
            abcdk_stcp_trace_output(node, LOG_WARNING, "加载共享钥失败，无法创建MaskSSL环境(scheme=%d)。", node->cfg.ssl_scheme);
            return -1;
        }

        abcdk_maskssl_set_fd(node->maskssl_ssl, node->fd, 0);
    }
    else if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI_ON_SK)
    {
#ifdef HEADER_SSL_H
        if (!node->openssl_bio)
            node->openssl_bio = abcdk_openssl_BIO_s_MaskSSL_form_file(node->cfg.sk_key_cipher, node->cfg.sk_key_file);
        else
            return -16;

        if (!node->openssl_bio)
        {
            abcdk_stcp_trace_output(node, LOG_WARNING, "加载共享钥失败，无法创建OpenSSL环境(scheme=%d)。", node->cfg.ssl_scheme);
            return -1;
        }

        if (!node->openssl_ssl)
            node->openssl_ssl = abcdk_openssl_ssl_alloc(node->flag == ABCDK_STCP_FLAG_ACCPET ? node->from_listen->openssl_ctx : node->openssl_ctx);
        else
            return -16;

        if (!node->openssl_ssl)
        {
            abcdk_stcp_trace_output(node, LOG_WARNING, "内存或资源不足，无法创建SSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
            return -1;
        }

        abcdk_openssl_BIO_set_fd(node->openssl_bio, node->fd);
        SSL_set_bio(node->openssl_ssl, node->openssl_bio, node->openssl_bio);

        /*托管理给SSL，这里要清理野指针。*/
        node->openssl_bio = NULL;

        if (node->flag == ABCDK_STCP_FLAG_ACCPET)
            SSL_set_accept_state(node->openssl_ssl);
        else if (node->flag == ABCDK_STCP_FLAG_CLIENT)
            SSL_set_connect_state(node->openssl_ssl);
        else
            return -22;
#else
        abcdk_stcp_trace_output(node, LOG_WARNING, "构建时未包含相关组件，无法创建OpenSSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
        return -22;
#endif // HEADER_SSL_H
    }
    else
    {
        return -22;
    }

    return 0;
}

void _abcdk_stcp_handshake(abcdk_stcp_node_t *node)
{
    int ssl_chk;
    int ssl_err;
    int chk;

    if (node->status == ABCDK_STCP_STATUS_SYNC)
    {
        chk = abcdk_poll(node->fd, 0x02, 0);
        if (chk > 0)
        {
            /*初始化SSL方案。*/
            chk = _abcdk_stcp_handshake_ssl_init(node);
            if (chk != 0)
                goto ERR;

            if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI || node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI_ON_SK)
                node->status = ABCDK_STCP_STATUS_SYNC_PKI;
            else if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_SK)
                node->status = ABCDK_STCP_STATUS_STABLE;
            else
                node->status = ABCDK_STCP_STATUS_STABLE;
        }
        else
        {
            chk = abcdk_asio_mark(node->asio_ctx, node->pfd, ABCDK_EPOLL_OUTPUT, 0);
            if (chk != 0)
                goto ERR;
            else
                goto END;
        }

        /*获取连接信息并设置默认值。*/
        _abcdk_stcp_handshake_sync_after(node);
    }

    if (node->status == ABCDK_STCP_STATUS_SYNC_PKI)
    {
#ifdef HEADER_SSL_H
        ssl_chk = SSL_do_handshake(node->openssl_ssl);
        if (ssl_chk == 1)
        {
            chk = _abcdk_stcp_openssl_verify_result(node);
            if (chk != 0)
                goto ERR;

            node->status = ABCDK_STCP_STATUS_STABLE;
        }
        else
        {
            /*必须通过返回值获取出错码。*/
            ssl_err = SSL_get_error(node->openssl_ssl, ssl_chk);

            if (ssl_err == SSL_ERROR_WANT_READ)
            {
                chk = abcdk_asio_mark(node->asio_ctx, node->pfd, ABCDK_EPOLL_INPUT, 0);
                if (chk == 0)
                    goto END;
            }
            else if (ssl_err == SSL_ERROR_WANT_WRITE)
            {
                chk = abcdk_asio_mark(node->asio_ctx, node->pfd, ABCDK_EPOLL_OUTPUT, 0);
                if (chk == 0)
                    goto END;
            }
            else
            {
                /*其它的全部当作出错处理。*/
                _abcdk_stcp_openssl_dump_errmsg(node, ssl_err);
            }

            /*Error .*/
            goto ERR;
        }
#endif // HEADER_SSL_H
    }

END:

    /*OK or AGAIN.*/
    return;

ERR:

    /*修改超时，使用超时检测器关闭。*/
    abcdk_asio_timeout(node->asio_ctx, node->pfd, -1);
}

static void _abcdk_stcp_dispatch(abcdk_stcp_t *ctx, uint32_t event, abcdk_stcp_node_t *node)
{
    abcdk_stcp_node_t *node_p;
    int chk;

    if (event & ABCDK_EPOLL_ERROR)
    {
        /*清除状态。*/
        node->status = 0;

        _abcdk_stcp_event_cb(node, ABCDK_STCP_EVENT_CLOSE, &chk);

        /*释放事件计数。*/
        abcdk_asio_unref(node->asio_ctx, node->pfd, ABCDK_EPOLL_ERROR);

        /*解除绑定关系。*/
        abcdk_asio_detch(node->asio_ctx, node->pfd);

        /*释放引用后，指针会被清空，因此这里需要复制一下。*/
        node_p = node;

        /*释放引用(关联时的引用)。*/
        abcdk_stcp_unref(&node_p);
    }

    if (event & ABCDK_EPOLL_OUTPUT)
    {
        if (node->status != ABCDK_STCP_STATUS_STABLE)
        {
            _abcdk_stcp_handshake(node);
            if (node->status == ABCDK_STCP_STATUS_STABLE)
                _abcdk_stcp_event_cb(node, ABCDK_STCP_EVENT_CONNECT, &chk);
        }
        else
        {
            _abcdk_stcp_event_cb(node, ABCDK_STCP_EVENT_OUTPUT, &chk);
        }

        /*无论连接状态如何，写权利必须内部释放，不能开放给应用层。*/
        abcdk_asio_mark(node->asio_ctx, node->pfd, 0, ABCDK_EPOLL_OUTPUT);

        /*释放事件计数。*/
        abcdk_asio_unref(node->asio_ctx, node->pfd, ABCDK_EPOLL_OUTPUT);
    }

    if (event & ABCDK_EPOLL_INPUT)
    {
        if (node->flag == ABCDK_STCP_FLAG_LISTEN)
        {
            /*每次处理一个新连接。*/
            _abcdk_stcp_accept(node);

            /*释放监听权利，并注册监听事件。*/
            abcdk_asio_mark(node->asio_ctx, node->pfd, ABCDK_EPOLL_INPUT, ABCDK_EPOLL_INPUT);
        }
        else
        {
            if (node->status != ABCDK_STCP_STATUS_STABLE)
            {
                _abcdk_stcp_handshake(node);
                if (node->status == ABCDK_STCP_STATUS_STABLE)
                    _abcdk_stcp_event_cb(node, ABCDK_STCP_EVENT_CONNECT, &chk);
            }
            else
            {
                _abcdk_stcp_event_cb(node, ABCDK_STCP_EVENT_INPUT, &chk);
            }

            /*无论连接状态如何，读权利必须内部释放，不能开放给应用层。*/
            abcdk_asio_mark(node->asio_ctx, node->pfd, 0, ABCDK_EPOLL_INPUT);
        }

        /*释放事件计数。*/
        abcdk_asio_unref(node->asio_ctx, node->pfd, ABCDK_EPOLL_INPUT);
    }
}

static void _abcdk_stcp_perform(abcdk_stcp_t *ctx, int idx)
{
    abcdk_asio_t *asio_ctx;
    abcdk_stcp_node_t *node;
    abcdk_epoll_event_t e;
    int chk;

    asio_ctx = abcdk_asioex_dispatch(ctx->asioex_ctx, idx);

    while (1)
    {
        memset(&e, 0, sizeof(abcdk_epoll_event_t));

        chk = abcdk_asio_wait(asio_ctx, &e);
        if (chk <= 0)
            break;

        node = (abcdk_stcp_node_t *)e.data.ptr;

        _abcdk_stcp_dispatch(ctx, e.events, node);
    }
}

static void _abcdk_stcp_worker(void *opaque, uint64_t event, void *item)
{
    abcdk_stcp_t *ctx = (abcdk_stcp_t *)opaque;

    _abcdk_stcp_perform(ctx, event);
}

void abcdk_stcp_stop(abcdk_stcp_t **ctx)
{
    _abcdk_stcp_ctx_unref(ctx);
}

abcdk_stcp_t *abcdk_stcp_start(int worker)
{
    return _abcdk_stcp_ctx_alloc(worker);
}

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
static int _abcdk_stcp_openssl_alpn_select_cb(SSL *ssl, const unsigned char **out, unsigned char *outlen,
                                              const unsigned char *in, unsigned int inlen, void *arg)
{
    abcdk_stcp_node_t *node_p;
    const unsigned char *srv;
    unsigned int srvlen;

    node_p = (abcdk_stcp_node_t *)arg;

    if (!node_p->cfg.pki_next_proto)
        return SSL_TLSEXT_ERR_ALERT_FATAL;

    srv = node_p->cfg.pki_next_proto;
    srvlen = strlen(node_p->cfg.pki_next_proto);

    /*服务端在客户端支持的协议列表中选择一个支持协议，从左到右按顺序匹配。*/
    if (SSL_select_next_proto((unsigned char **)out, outlen, in, inlen, srv, srvlen) != OPENSSL_NPN_NEGOTIATED)
    {
        return SSL_TLSEXT_ERR_ALERT_FATAL;
    }

    return SSL_TLSEXT_ERR_OK;
}
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H

static void _abcdk_stcp_openssl_set_alpn(abcdk_stcp_node_t *node)
{

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
    SSL_CTX_set_alpn_select_cb(node->openssl_ctx, _abcdk_stcp_openssl_alpn_select_cb, (void *)node);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H
}

static int _abcdk_stcp_ssl_init(abcdk_stcp_node_t *node, int listen_flag)
{
    int ssl_chk;
    int ssl_err;
    int chk;

    if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI)
    {
#ifdef HEADER_SSL_H
        node->openssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(listen_flag, (node->cfg.pki_check_cert ? node->cfg.pki_ca_file : NULL),
                                                             (node->cfg.pki_check_cert ? node->cfg.pki_ca_path : NULL),
                                                             node->cfg.pki_cert_file, node->cfg.pki_key_file, NULL);

        if (!node->openssl_ctx)
        {
            abcdk_stcp_trace_output(node, LOG_WARNING, "加载证书或私钥失败，无法创建SSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
            return -2;
        }

        /*设置密码套件。*/
        if (node->cfg.pki_cipher_list)
        {
            ssl_chk = SSL_CTX_set_cipher_list(node->openssl_ctx, node->cfg.pki_cipher_list);
            if (ssl_chk != 1)
            {
                ssl_err = SSL_get_error(node->openssl_ssl, ssl_chk);
                _abcdk_stcp_openssl_dump_errmsg(node, ssl_err);
                return -3;
            }
        }

        /*设置下层协议。*/
        if (node->cfg.pki_next_proto)
            _abcdk_stcp_openssl_set_alpn(node);
#else
        abcdk_stcp_trace_output(node, LOG_WARNING, "构建时未包含相关组件，无法创建OpenSSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
        return -22;
#endif // HEADER_SSL_H
    }
    else if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_SK)
    {
        node->maskssl_ssl = abcdk_maskssl_create_from_file(node->cfg.sk_key_cipher, node->cfg.sk_key_file);
        if (!node->maskssl_ssl)
        {
            abcdk_stcp_trace_output(node, LOG_WARNING, "加载共享钥失败，无法创建MaskSSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
            return -2;
        }

        /*仅用于验证。*/
        abcdk_maskssl_destroy(&node->maskssl_ssl);
    }
    else if (node->cfg.ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI_ON_SK)
    {
#ifdef HEADER_SSL_H
        node->openssl_bio = abcdk_openssl_BIO_s_MaskSSL_form_file(node->cfg.sk_key_cipher, node->cfg.sk_key_file);
        if (!node->openssl_bio)
        {
            abcdk_stcp_trace_output(node, LOG_WARNING, "加载共享钥失败，无法创建OpenSSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
            return -2;
        }

        /*仅用于验证。*/
        abcdk_openssl_BIO_destroy(&node->openssl_bio);

        node->openssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(listen_flag, (node->cfg.pki_check_cert ? node->cfg.pki_ca_file : NULL),
                                                             (node->cfg.pki_check_cert ? node->cfg.pki_ca_path : NULL),
                                                             node->cfg.pki_cert_file, node->cfg.pki_key_file, NULL);

        if (!node->openssl_ctx)
        {
            abcdk_stcp_trace_output(node, LOG_WARNING, "加载证书或私钥失败，无法创建OpenSSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
            return -2;
        }

        /*设置密码套件。*/
        if (node->cfg.pki_cipher_list)
        {
            ssl_chk = SSL_CTX_set_cipher_list(node->openssl_ctx, node->cfg.pki_cipher_list);
            if (ssl_chk != 1)
            {
                ssl_err = SSL_get_error(node->openssl_ssl, ssl_chk);
                _abcdk_stcp_openssl_dump_errmsg(node, ssl_err);
                return -3;
            }
        }

        /*设置下层协议。*/
        _abcdk_stcp_openssl_set_alpn(node);

#else
        abcdk_stcp_trace_output(node, LOG_WARNING, "构建时未包含相关组件，无法创建OpenSSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
        return -22;
#endif // HEADER_SSL_H
    }

    return 0;
}

static void _abcdk_stcp_fix_cfg(abcdk_stcp_node_t *node)
{
    /*修复不支持的配置和默认值。*/

    node->cfg.sk_key_file = (node->cfg.sk_key_file ? node->cfg.sk_key_file : "");

    if(node->cfg.sk_key_cipher == 0)
        node->cfg.sk_key_cipher = ABCDK_MASKSSL_SCHEME_ENIGMA;

    if (node->cfg.out_hook_min_th <= 0)
        node->cfg.out_hook_min_th = 200;
    else
        node->cfg.out_hook_min_th = ABCDK_CLAMP(node->cfg.out_hook_min_th, 200, 600);

    if (node->cfg.out_hook_max_th <= 0)
        node->cfg.out_hook_max_th = 400;
    else
        node->cfg.out_hook_max_th = ABCDK_CLAMP(node->cfg.out_hook_max_th, 400, 800);

    if (node->cfg.out_hook_weight <= 0)
        node->cfg.out_hook_weight = 2;
    else
        node->cfg.out_hook_weight = ABCDK_CLAMP(node->cfg.out_hook_weight, 1, 99);

    if (node->cfg.out_hook_prob <= 0)
        node->cfg.out_hook_prob = 2;
    else
        node->cfg.out_hook_prob = ABCDK_CLAMP(node->cfg.out_hook_prob, 1, 99);

    /*最小阈值和最大阈值必须符合区间要求。*/
    if (node->cfg.out_hook_min_th > node->cfg.out_hook_max_th)
        ABCDK_INTEGER_SWAP(node->cfg.out_hook_min_th, node->cfg.out_hook_max_th);
}

int abcdk_stcp_listen(abcdk_stcp_node_t *node, abcdk_sockaddr_t *addr, abcdk_stcp_config_t *cfg)
{
    abcdk_stcp_node_t *node_p = NULL;
    epoll_data_t ep_data;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && addr != NULL && cfg != NULL);
    ABCDK_ASSERT(cfg->prepare_cb != NULL, "未绑定通知回调函数，通讯对象无法正常工作。");
    ABCDK_ASSERT(cfg->event_cb != NULL, "未绑定通知回调函数，通讯对象无法正常工作。");

    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_stcp_refer(node);

    node_p->flag = ABCDK_STCP_FLAG_LISTEN;
    node_p->status = ABCDK_STCP_STATUS_STABLE;
    node_p->cfg = *cfg;

    /*修复不支持的配置。*/
    _abcdk_stcp_fix_cfg(node_p);

    /*UNIX需要特殊复制一下。*/
    if (addr->family == AF_UNIX)
    {
        node_p->local.family = AF_UNIX;
        strcpy(node_p->local.addr_un.sun_path, addr->addr_un.sun_path);
    }
    else
    {
        node_p->local = *addr;
    }

    node_p->fd = abcdk_socket(node_p->local.family, 0);
    if (node_p->fd < 0)
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

    if (addr->family == AF_INET6)
    {
        /*IPv6仅支持IPv6。*/
        sock_flag = 1;
        chk = abcdk_sockopt_option_int(node_p->fd, IPPROTO_IPV6, IPV6_V6ONLY, &sock_flag, 2);
        if (chk != 0)
            goto ERR;
    }

    chk = abcdk_bind(node_p->fd, &node_p->local);
    if (chk != 0)
        goto ERR;

    chk = listen(node_p->fd, SOMAXCONN);
    if (chk != 0)
        goto ERR;

    chk = abcdk_fflag_add(node_p->fd, O_NONBLOCK);
    if (chk != 0)
        goto ERR;

    chk = _abcdk_stcp_ssl_init(node_p, 1);
    if (chk != 0)
        goto ERR;

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

    abcdk_stcp_unref(&node_p);

    return -1;
}

int abcdk_stcp_connect(abcdk_stcp_node_t *node, abcdk_sockaddr_t *addr, abcdk_stcp_config_t *cfg)
{
    abcdk_stcp_node_t *node_p = NULL;
    epoll_data_t ep_data;
    socklen_t addr_len;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && addr != NULL && cfg != NULL);
    ABCDK_ASSERT(cfg->event_cb != NULL, "未绑定通知回调函数，通讯对象无法正常工作。");

    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_stcp_refer(node);

    node_p->flag = ABCDK_STCP_FLAG_CLIENT;
    node_p->status = ABCDK_STCP_STATUS_SYNC;
    node_p->cfg = *cfg;

    /*修复不支持的配置。*/
    _abcdk_stcp_fix_cfg(node_p);

    addr_len = sizeof(abcdk_sockaddr_t);
    if (addr->family == AF_UNIX)
    {
        addr_len = SUN_LEN(&addr->addr_un);
        node_p->remote.family = AF_UNIX;
        strcpy(node_p->remote.addr_un.sun_path, addr->addr_un.sun_path);
    }
    else if (addr->family == AF_INET)
    {
        addr_len = sizeof(struct sockaddr_in);
        node_p->remote = *addr;
    }
    else if (addr->family == AF_INET6)
    {
        addr_len = sizeof(struct sockaddr_in6);
        node_p->remote = *addr;
    }

    node_p->fd = abcdk_socket(node_p->remote.family, 0);
    if (node_p->fd < 0)
        goto ERR;

    chk = abcdk_fflag_add(node_p->fd, O_NONBLOCK);
    if (chk != 0)
        goto ERR;

    chk = connect(node_p->fd, &node_p->remote.addr, addr_len);
    if (chk != 0 && errno != EAGAIN && errno != EINPROGRESS)
        goto ERR;

    chk = _abcdk_stcp_ssl_init(node_p, 0);
    if (chk != 0)
        goto ERR;

    node_p->out_wred = abcdk_wred_create(node_p->cfg.out_hook_min_th, node_p->cfg.out_hook_max_th,
                                         node_p->cfg.out_hook_weight, node_p->cfg.out_hook_prob);
    if (!node_p->out_wred)
        goto ERR;

    node_p->asio_ctx = abcdk_asioex_dispatch(node_p->ctx->asioex_ctx, -1);
    if (!node_p->asio_ctx)
        goto ERR;

    /*节点加入epoll池中。在解除绑定关系前，节点不会被释放。*/
    ep_data.ptr = node_p;
    node_p->pfd = abcdk_asio_attach(node_p->asio_ctx, node_p->fd, &ep_data);
    if (node_p->pfd <= 0)
        goto ERR;

    abcdk_asio_timeout(node_p->asio_ctx, node_p->pfd, 180);
    abcdk_asio_mark(node_p->asio_ctx, node_p->pfd, ABCDK_EPOLL_OUTPUT, 0);

    return 0;

ERR:

    abcdk_stcp_unref(&node_p);

    return -1;
}

static void _abcdk_stcp_input_hook(abcdk_stcp_node_t *node)
{
    ssize_t rlen = 0, pos = 0;
    size_t remain = 0;
    int chk = 0;

    /*当未注册输入数据到达通知回调函数时，直接发事件通知。*/
    if (!node->cfg.input_cb)
    {
        node->cfg.event_cb(node, ABCDK_STCP_EVENT_INPUT, &chk);
        return;
    }

    if (!node->in_buffer)
    {
        node->in_buffer = abcdk_object_alloc2(64 * 1024);
        if (!node->in_buffer)
        {
            abcdk_stcp_set_timeout(node, -1);
            return;
        }
    }

NEXT_MSG:

    /*清零.*/
    rlen = pos = 0;
    remain = 0;

    /*收。*/
    rlen = abcdk_stcp_recv(node, node->in_buffer->pptrs[0], node->in_buffer->sizes[0]);
    if (rlen <= 0)
    {
        abcdk_stcp_recv_watch(node);
        return;
    }

    /*缓存中可能存在多个请求，因此处理所有请求才能退出循环。*/
    while (pos < rlen)
    {
        node->cfg.input_cb(node, ABCDK_PTR2VPTR(node->in_buffer->pptrs[0], pos), rlen - pos, &remain);
        pos += (rlen - pos) - remain;
    }

    /*继续读取缓存内可能存在数据，直到为空。*/
    goto NEXT_MSG;
}

static void _abcdk_stcp_output_hook(abcdk_stcp_node_t *node)
{
    abcdk_tree_t *p;
    ssize_t slen = 0;
    int chk;

NEXT_MSG:

    /*从队列头部开始发送。*/
    abcdk_spinlock_lock(node->out_locker, 1);
    p = abcdk_tree_child(node->out_queue, 1);
    abcdk_spinlock_unlock(node->out_locker);

    /*通知应用层，发送队列空闲。*/
    if (!p)
    {
        node->cfg.event_cb(node, ABCDK_STCP_EVENT_OUTPUT, &chk);
        return;
    }

    /*
     * 发。
     * 
     * 注1：能发多少发多少。
     * 注2：重发时，数据的参数不能改变(指针和长度)。
     */
    while (node->out_pos < p->obj->sizes[0])
    {
        slen = abcdk_stcp_send(node, ABCDK_PTR2VPTR(p->obj->pptrs[0], node->out_pos), p->obj->sizes[0] - node->out_pos);
        if (slen <= 0)
        {
            abcdk_stcp_send_watch(node);
            return;
        }

        /*累加已发送的长度。*/
        node->out_pos += slen;
    }

    /*代码走到这里，表示当前节点的数据已经全部发出，因此游标归零。*/
    node->out_pos = 0;

    /*移除节点。*/
    abcdk_spinlock_lock(node->out_locker, 1);
    abcdk_tree_unlink(p);
    node->out_len -= 1;
    abcdk_spinlock_unlock(node->out_locker);

    /*删除节点。*/
    abcdk_tree_free(&p);

    goto NEXT_MSG;
}


int abcdk_stcp_post(abcdk_stcp_node_t *node, abcdk_object_t *data, int key)
{
    abcdk_tree_t *p;
    int chk;

    assert(node != NULL && data != NULL);
    assert(data->pptrs[0] != NULL && data->sizes[0] > 0);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if (!node->flag || !node->status)
        return -3;

    if (node->flag == ABCDK_STCP_FLAG_LISTEN)
        return -2;

    p = abcdk_tree_alloc(data);
    if (!p)
        return -1;

    abcdk_spinlock_lock(node->out_locker, 1);

    /*非关键数据根据WRED算法决定是否添加到队列中。*/
    chk = key ? 0 : abcdk_wred_update(node->out_wred, node->out_len + 1);
    if (chk == 0)
    {
        abcdk_tree_insert2(node->out_queue, p, 0);
        node->out_len += 1;
    }
    else
    {
        abcdk_stcp_trace_output(node, LOG_WARNING, "输出缓慢，队列积压过长(len=%d)，丢弃当前数据包(size=%zd)。\n", node->out_len, p->obj->sizes[0]);

        abcdk_tree_free(&p);
    }

    abcdk_spinlock_unlock(node->out_locker);

    if (node->status == ABCDK_STCP_STATUS_STABLE)
        abcdk_stcp_send_watch(node);

    return 0;
}

int abcdk_stcp_post_buffer(abcdk_stcp_node_t *node, const void *data, size_t size)
{
    abcdk_object_t *obj;
    int chk;

    assert(node != NULL && data != NULL && size > 0);

    obj = abcdk_object_copyfrom(data, size);
    if (!obj)
        return -1;

    chk = abcdk_stcp_post(node, obj, 1);
    if (chk == 0)
        return 0;

    /*删除投递失败的。*/
    abcdk_object_unref(&obj);
    return chk;
}

int abcdk_stcp_post_vformat(abcdk_stcp_node_t *node, int max, const char *fmt, va_list ap)
{
    abcdk_object_t *obj;
    int chk;

    assert(node != NULL && max > 0 && fmt != NULL);

    obj = abcdk_object_vprintf(max, fmt, ap);
    if (!obj)
        return -1;

    chk = abcdk_stcp_post(node, obj, 1);
    if (chk == 0)
        return 0;

    /*删除投递失败的。*/
    abcdk_object_unref(&obj);
    return chk;
}

int abcdk_stcp_post_format(abcdk_stcp_node_t *node, int max, const char *fmt, ...)
{
    int chk;

    assert(node != NULL && max > 0 && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    chk = abcdk_stcp_post_vformat(node, max, fmt, ap);
    va_end(ap);

    return chk;
}