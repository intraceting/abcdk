/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/asio/asio.h"

/** 简单的异步TCP通讯。 */
struct _abcdk_asio
{
    /** epollex 环境。*/
    abcdk_epollex_t *epollex;

    /** 工作线程。*/
    abcdk_thread_t worker;

    /** 最大连接数量。*/
    int max;

    /** 退出标志。0： 运行，!0：停止。*/
    volatile int exitflag;

};// abcdk_asio_t;

/** 异步TCP节点。 */
struct _abcdk_asio_node
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_ASIO_NODE_MAGIC 123456789

    /** 引用计数器。*/
    volatile int refcount;

    /** 
     * 通讯环境指针。
     * 
     * @note 仅复制。
    */
    abcdk_asio_t *ctx;

    /** 配置。*/
    abcdk_asio_config_t cfg;

    /**
     * 索引。
     */
    uint64_t index;

    /** 标识句柄来源。*/
    volatile int flag;
#define ABCDK_ASIO_FLAG_CLIENT   1
#define ABCDK_ASIO_FLAG_LISTEN   2
#define ABCDK_ASIO_FLAG_ACCPET   3

    /** 标识当前句柄状态。*/
    volatile int status;
#define ABCDK_ASIO_STATUS_STABLE        1
#define ABCDK_ASIO_STATUS_SYNC          2
#define ABCDK_ASIO_STATUS_SYNC_OPENSSL  3

    /** 本机地址。*/
    abcdk_sockaddr_t local;

    /** 远端地址。*/
    abcdk_sockaddr_t remote;

    /** 句柄。*/
    int fd;

    /** openssl环境指针。*/
    SSL_CTX *openssl_ctx;

    /** openssl环境指针。*/
    SSL *openssl_ssl;
    BIO *openssl_bio;

    /** easyssl环境指针。*/
    abcdk_easyssl_t *easyssl_ssl;

    /** 工作线程。*/
    volatile pthread_t worker;


    /** 用户环境指针。*/
    abcdk_object_t *userdata;

    /** 用户环境销毁函数。*/
    void (*userdata_free_cb)(void *userdata);

    /** 发送队列。*/
    abcdk_tree_t *out_queue;

    /** 发送队列锁。*/
    abcdk_mutex_t *out_locker;

    /** 发送游标。*/
    size_t out_pos;

    /** 接收缓存。*/
    abcdk_object_t *in_buffer;

    /** 来自哪个监听节点。*/
    abcdk_asio_node_t *from_listen;


};// abcdk_asio_node_t;

void abcdk_asio_unref(abcdk_asio_node_t **node)
{
    abcdk_asio_node_t *node_p = NULL;

    if (!node || !*node)
        return;

    node_p = *node;
    *node = NULL;

    assert(node_p->magic == ABCDK_ASIO_NODE_MAGIC);

    if (abcdk_atomic_fetch_and_add(&node_p->refcount, -1) != 1)
        return;

    assert(node_p->refcount == 0);

    node_p->magic = 0xcccccccc;

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_free(&node_p->openssl_ssl);
    abcdk_BIO_destroy(&node_p->openssl_bio);
    abcdk_openssl_ssl_ctx_free(&node_p->openssl_ctx);
#endif //HEADER_SSL_H

    abcdk_easyssl_destroy(&node_p->easyssl_ssl);

    /*直接关闭，快速回收资源，不会处于time_wait状态。*/
    if (node_p->fd >= 0)
    {
        struct linger l = {1, 0};
        abcdk_socket_option_linger(node_p->fd, &l, 2);
    }

    if(node_p->userdata_free_cb)
        node_p->userdata_free_cb(node_p->userdata->pptrs[0]);

    abcdk_closep(&node_p->fd);
    abcdk_object_unref(&node_p->userdata);
    abcdk_tree_free(&node_p->out_queue);
    abcdk_mutex_destroy(&node_p->out_locker);
    abcdk_object_unref(&node_p->in_buffer);
    abcdk_asio_unref(&node_p->from_listen);
    abcdk_heap_free(node_p);
}

abcdk_asio_node_t *abcdk_asio_refer(abcdk_asio_node_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_asio_node_t *abcdk_asio_alloc(abcdk_asio_t *ctx,size_t userdata, void (*free_cb)(void *userdata))
{
    abcdk_asio_node_t *node = NULL;

    assert(ctx != NULL);

    node = (abcdk_asio_node_t *)abcdk_heap_alloc(sizeof(abcdk_asio_node_t));
    if(!node)
        return NULL;

    node->magic = ABCDK_ASIO_NODE_MAGIC;
    node->refcount = 1;
    node->ctx = ctx;
    node->index = abcdk_sequence_num();
    node->fd = -1;
    node->userdata = abcdk_object_alloc3(userdata,1);
    node->userdata_free_cb = free_cb;
    node->out_queue = abcdk_tree_alloc3(1);
    node->out_locker = abcdk_mutex_create();
    node->out_pos = 0;
    node->in_buffer = abcdk_object_alloc2(256*1024);
    node->from_listen = NULL;
    node->openssl_ctx = NULL;
    node->openssl_ssl = NULL;
    node->openssl_bio = NULL;
    node->easyssl_ssl = NULL;

    return node;
}

void abcdk_asio_trace_output(abcdk_asio_node_t *node,int type, const char* fmt,...)
{
    char new_tname[18] = {0}, old_tname[18] = {0};

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

uint64_t abcdk_asio_get_index(abcdk_asio_node_t *node)
{
    assert(node != NULL);

    return node->index;
}

SSL *abcdk_asio_openssl_get_handle(abcdk_asio_node_t *node)
{
    assert(node != NULL);

    return node->openssl_ssl;
}

char *abcdk_asio_openssl_get_alpn_selected(abcdk_asio_node_t *node, char proto[255+1])
{
    int chk;

    assert(node != NULL && proto != NULL);

    if(!node->openssl_ssl)
        return NULL;

#ifdef HEADER_SSL_H

    chk = abcdk_openssl_ssl_get_alpn_selected(node->openssl_ssl, proto);
    if(chk != 0)
        return proto;

#endif // HEADER_SSL_H

    return NULL;
}

void *abcdk_asio_get_userdata(abcdk_asio_node_t *node)
{
    assert(node != NULL);

    return node->userdata->pptrs[0];
}

void *abcdk_asio_set_userdata(abcdk_asio_node_t *node,void *opaque)
{
    void *old;

    assert(node != NULL);

    old = node->userdata->pptrs[0];
    node->userdata->pptrs[0] = (uint8_t*)opaque;

    return old;
}

int abcdk_asio_set_timeout(abcdk_asio_node_t *node, time_t timeout)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if(!node->flag ||!node->status)
        return -3;

    chk = abcdk_epollex_timeout(node->ctx->epollex, node->fd, timeout);

    return chk;
}

int abcdk_asio_get_sockaddr(abcdk_asio_node_t *node, abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote)
{
    assert(node != NULL);

    if (local && node->local.family)
        abcdk_sockaddr_copy(&node->local,local);

    if (remote && node->remote.family)
        abcdk_sockaddr_copy(&node->remote,remote);

    return 0;
}

int abcdk_asio_get_sockaddr_str(abcdk_asio_node_t *node, char local[NAME_MAX],char remote[NAME_MAX])
{
    assert(node != NULL);

    if(local && node->local.family)
        abcdk_sockaddr_to_string(local,&node->local);

    if(remote && node->remote.family)
        abcdk_sockaddr_to_string(remote,&node->remote);

    return 0;
}

ssize_t abcdk_asio_recv(abcdk_asio_node_t *node, void *buf, size_t size)
{
    ssize_t rsize = 0,rsize_all = 0;
    int chk;

    assert(node != NULL && buf != NULL && size >0);

    /*仅消息循环线程拥有读权利。*/
    chk = abcdk_thread_leader_test(&node->worker);
    ABCDK_ASSERT(chk == 0,"当前线程没有读权利。");

    while (rsize_all < size)
    {
        if(node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI || node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
            rsize = SSL_read(node->openssl_ssl,ABCDK_PTR2PTR(void,buf,rsize_all),size-rsize_all);
        else if(node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
            rsize = abcdk_easyssl_read(node->easyssl_ssl,ABCDK_PTR2PTR(void,buf,rsize_all),size-rsize_all);
        else 
            rsize = read(node->fd,ABCDK_PTR2PTR(void,buf,rsize_all),size-rsize_all);
        
        if(rsize <=0)
            break;
        
        rsize_all += rsize;
    }

    return rsize_all;
}

int abcdk_asio_recv_watch(abcdk_asio_node_t *node)
{
    int done_flag = 0;
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if(!node->flag ||!node->status)
        return -3;

    /*仅允许拥有读权利的线程释放读权利，其它线程只能注册读事件。*/
    chk = abcdk_thread_leader_test(&node->worker);
    if (chk == 0)
        done_flag = ABCDK_EPOLL_INPUT;

    chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_INPUT, done_flag);

    return chk;
}

ssize_t abcdk_asio_send(abcdk_asio_node_t *node, void *buf, size_t size)
{
    ssize_t wsize = 0,wsize_all = 0;
    int chk;

    assert(node != NULL && buf != NULL && size > 0);

    /*仅消息循环线程拥有写权利。*/
    chk = abcdk_thread_leader_test(&node->worker);
    ABCDK_ASSERT(chk == 0,"当前线程没有写权利。");

    while (wsize_all < size)
    {
        if(node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI || node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
            wsize = SSL_write(node->openssl_ssl,ABCDK_PTR2PTR(void,buf,wsize_all),size-wsize_all);
        else if(node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
            wsize = abcdk_easyssl_write(node->easyssl_ssl,ABCDK_PTR2PTR(void,buf,wsize_all),size-wsize_all);
        else 
            wsize = write(node->fd,ABCDK_PTR2PTR(void,buf,wsize_all),size-wsize_all);
        
        if(wsize <=0)
            break;
        
        wsize_all += wsize;
    }

    return wsize_all;
}

int abcdk_asio_send_watch(abcdk_asio_node_t *node)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if(!node->flag ||!node->status)
        return -3;

    chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_OUTPUT, 0);

    return chk;
}

void _abcdk_asio_cleanup_cb(epoll_data_t *data, void *opaque)
{
    abcdk_asio_t *ctx = (abcdk_asio_t *)opaque;
    abcdk_asio_node_t *node = NULL;

    node = (abcdk_asio_node_t *)data->ptr;
    abcdk_asio_unref(&node);
}

void _abcdk_asio_prepare_cb(abcdk_asio_node_t **node, abcdk_asio_node_t *listen)
{
    /*通知应用层处理事件。*/
    if (listen->cfg.prepare_cb)
        listen->cfg.prepare_cb(node, listen);
}

/*声明输入事件钩子函数。*/
void _abcdk_asio_input_hook(abcdk_asio_node_t *node);

/*声明输出事件钩子函数。*/
void _abcdk_asio_output_hook(abcdk_asio_node_t *node);

void _abcdk_asio_event_cb(abcdk_asio_node_t *node,uint32_t event, int *result)
{
    /*绑定工作线程。*/
    abcdk_thread_leader_vote(&node->worker);

    /*通知应用层处理事件。*/
    if(event == ABCDK_ASIO_EVENT_INPUT)
        _abcdk_asio_input_hook(node);
    else if(event == ABCDK_ASIO_EVENT_OUTPUT)
        _abcdk_asio_output_hook(node);
    else 
        node->cfg.event_cb(node, event, result);

    /*解绑工作线程。*/
    abcdk_thread_leader_quit(&node->worker);
}

void _abcdk_asio_accept(abcdk_asio_node_t *listen)
{
    abcdk_asio_node_t *node = NULL;
    epoll_data_t ep_data;
    int chk;

    /*通知初始化。*/
    _abcdk_asio_prepare_cb(&node, listen);
    if (!node)
        return;

    /*配置参数。*/
    node->flag = ABCDK_ASIO_FLAG_ACCPET;
    node->status = ABCDK_ASIO_STATUS_SYNC;

    /*记住来源。*/
    node->from_listen = abcdk_asio_refer(listen);

    /*复制通讯环境指针。*/
    node->ctx = listen->ctx;

    /*复制监听环境的配置。*/
    node->cfg = listen->cfg;

    /*每次取出一个句柄。*/
    node->fd = abcdk_accept(listen->fd, &node->remote);
    if (node->fd < 0)
        goto final_error;
    
    /*
     * 检测最大连接数量限制。
     *
     * 如果不把已经建立的连接从监听队列除，那么新的连接可能无法连接。
    */
    if(abcdk_epollex_count(node->ctx->epollex) >= node->ctx->max)
        goto final_error;

    /*通知应用层新连接到来。*/
    _abcdk_asio_event_cb(node,ABCDK_ASIO_EVENT_ACCEPT,&chk);
    if(chk != 0 )
        goto final_error;
    
    chk = abcdk_fflag_add(node->fd,O_NONBLOCK);
    if(chk != 0 )
        goto final_error;

    ep_data.ptr = node;
    chk = abcdk_epollex_attach(node->ctx->epollex, node->fd, &ep_data);
    if(chk != 0)
        goto final_error;

    abcdk_epollex_timeout(node->ctx->epollex, node->fd, 180*1000);
    
    /*注册输出事件用于探测连接状态。*/
    abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_OUTPUT, 0);
    

    return;

final_error:

    /*通知关闭。*/
    _abcdk_asio_event_cb(node, ABCDK_ASIO_EVENT_INTERRUPT,&chk);
    abcdk_asio_unref(&node);
    
    return;
}

#ifdef HEADER_SSL_H   
static int _abcdk_asio_openssl_verify_result(abcdk_asio_node_t *node)
{
    char remote_addr[NAME_MAX] = {0};
    int chk;

    abcdk_asio_get_sockaddr_str(node, NULL, remote_addr);

    X509 *cert = SSL_get_peer_certificate(node->openssl_ssl);
    if (cert)
    {
        abcdk_object_t *info = abcdk_openssl_dump_crt(cert);
        if (info)
        {
            abcdk_asio_trace_output(node, LOG_INFO, "远端(%s)的证书信息：\n%s", remote_addr, info->pstrs[0]);
            abcdk_object_unref(&info);
        }

        X509_free(cert);
    }

    if (node->cfg.pki_check_cert)
    {
        chk = SSL_get_verify_result(node->openssl_ssl);
        if (chk != X509_V_OK)
        {
            abcdk_asio_trace_output(node, LOG_INFO, "远端(%s)的证书验证有错误发生(ssl-errno=%d)。", remote_addr, chk);
            return -1;
        }
    }

    return 0;
}

static void _abcdk_asio_openssl_dump_errmsg(abcdk_asio_node_t *node, unsigned long e)
{
    char remote_addr[NAME_MAX] = {0};
    char local_addr[NAME_MAX] = {0};
    char errmsg[NAME_MAX] = {0};

    ERR_error_string_n(e,errmsg,NAME_MAX-1);

    abcdk_asio_get_sockaddr_str(node, local_addr, remote_addr);

    abcdk_asio_trace_output(node, LOG_INFO, "本机(%s)与远端(%s)的连接有错误发生(%s)。", local_addr, remote_addr,errmsg);
}

#endif //HEADER_SSL_H

static void _abcdk_asio_handshake_sync_after(abcdk_asio_node_t *node)
{
    socklen_t sock_len = 0;
    int sock_flag = 1;
    struct timeval tv;

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
    abcdk_sockopt_option_timeout(node->fd, SO_RCVTIMEO, &tv, 2);
    abcdk_sockopt_option_timeout(node->fd, SO_SNDTIMEO, &tv, 2);

    /*修改保活参数，以防在远程断电的情况下本地无法检测到连接断开信号。*/

    /*开启keepalive属性*/
    sock_flag = 1;
    abcdk_sockopt_option_int(node->fd, SOL_SOCKET, SO_KEEPALIVE, &sock_flag, 2);

    /*连接在60秒内没有任何数据往来，则进行探测。*/
    sock_flag = 60;
    abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_KEEPIDLE, &sock_flag, 2);

    /*探测时发包的时间间隔为5秒。*/
    sock_flag = 5;
    abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_KEEPINTVL, &sock_flag, 2);

    /*探测尝试的次数.如果第一次探测包就收到响应，则后两次的不再发。*/
    sock_flag = 3;
    abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_KEEPCNT, &sock_flag, 2);

    /*关闭延迟发送。*/
    sock_flag = 1;
    abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_NODELAY, &sock_flag, 2);
}

static int _abcdk_asio_handshake_ssl_init(abcdk_asio_node_t *node)
{
    if(node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_RAW)
    {
        return 0;
    }
    else if(node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI)
    {
#ifdef HEADER_SSL_H
        if(!node->openssl_ssl)
            node->openssl_ssl = abcdk_openssl_ssl_alloc(node->flag == ABCDK_ASIO_FLAG_ACCPET?node->from_listen->openssl_ctx:node->openssl_ctx);
        else 
            return -16;

        if(!node->openssl_ssl)
        {
            abcdk_asio_trace_output(node,LOG_WARNING, "内存或资源不足，无法创建SSL环境(ssl-scheme=%d)。",node->cfg.ssl_scheme);
            return -1;
        }

        SSL_set_fd(node->openssl_ssl, node->fd);

        if (node->flag == ABCDK_ASIO_FLAG_ACCPET)
            SSL_set_accept_state(node->openssl_ssl);
        else if (node->flag == ABCDK_ASIO_FLAG_CLIENT)
            SSL_set_connect_state(node->openssl_ssl);
        else
            return -22;
#else 
        abcdk_asio_trace_output(node,LOG_WARNING, "构建时未包含相关组件，无法创建SSL环境。");
        return -22;
#endif //HEADER_SSL_H
    }
    else if (node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
    {
        if(!node->easyssl_ssl)
            node->easyssl_ssl = abcdk_easyssl_create_from_file(node->cfg.enigma_key_file,ABCDK_EASYSSL_SCHEME_ENIGMA,node->cfg.enigma_salt_size);
        else
            return -16;
            
        if (!node->easyssl_ssl)
        {
            abcdk_asio_trace_output(node,LOG_WARNING, "加载共享钥失败，无法创建SSL环境(ssl-scheme=%d)。",node->cfg.ssl_scheme);
            return -1;
        }

        abcdk_easyssl_set_fd(node->easyssl_ssl, node->fd,0);
    }
    else if (node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
    {
#ifdef HEADER_SSL_H
        if (!node->openssl_bio)
            node->openssl_bio = abcdk_BIO_s_easyssl(node->cfg.enigma_key_file, ABCDK_EASYSSL_SCHEME_ENIGMA, node->cfg.enigma_salt_size);
        else
            return -16;

        if (!node->openssl_bio)
        {
            abcdk_asio_trace_output(node, LOG_WARNING, "加载共享钥失败，无法创建SSL环境(ssl-scheme=%d)。", node->cfg.ssl_scheme);
            return -1;
        }

        if(!node->openssl_ssl)
            node->openssl_ssl = abcdk_openssl_ssl_alloc(node->flag == ABCDK_ASIO_FLAG_ACCPET?node->from_listen->openssl_ctx:node->openssl_ctx);
        else 
            return -16;

        if(!node->openssl_ssl)
        {
            abcdk_asio_trace_output(node,LOG_WARNING, "内存或资源不足，无法创建SSL环境(ssl-scheme=%d)。",node->cfg.ssl_scheme);
            return -1;
        }

        abcdk_BIO_set_fd(node->openssl_bio,node->fd);
        SSL_set_bio(node->openssl_ssl, node->openssl_bio, node->openssl_bio);

        /*托管理给SSL，这里要清理野指针。*/
        node->openssl_bio = NULL;

        if (node->flag == ABCDK_ASIO_FLAG_ACCPET)
            SSL_set_accept_state(node->openssl_ssl);
        else if (node->flag == ABCDK_ASIO_FLAG_CLIENT)
            SSL_set_connect_state(node->openssl_ssl);
        else
            return -22;
#else 
        abcdk_asio_trace_output(node,LOG_WARNING, "构建时未包含相关组件，无法创建SSL环境。");
        return -22;
#endif //HEADER_SSL_H
    }
    else
    {
        return -22;
    }
    
    return 0;
}

void _abcdk_asio_handshake(abcdk_asio_node_t *node)
{
    int ssl_chk;
    int ssl_err;
    int chk;

    if (node->status == ABCDK_ASIO_STATUS_SYNC)
    {
        chk = abcdk_poll(node->fd, 0x02, 0);
        if (chk > 0)
        {
            /*初始化SSL方案。*/
            chk = _abcdk_asio_handshake_ssl_init(node);
            if(chk != 0)
                goto final_error;

            if(node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI || node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
                node->status = ABCDK_ASIO_STATUS_SYNC_OPENSSL;
            else if(node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
                node->status = ABCDK_ASIO_STATUS_STABLE;
            else 
                node->status = ABCDK_ASIO_STATUS_STABLE;
        }
        else
        {
            chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_OUTPUT, 0);
            if (chk != 0)
                goto final_error;
            else 
                goto final;
        }

        /*获取连接信息并设置默认值。*/
        _abcdk_asio_handshake_sync_after(node);
    }
     
    if (node->status == ABCDK_ASIO_STATUS_SYNC_OPENSSL)
    {
#ifdef HEADER_SSL_H 
        ssl_chk = SSL_do_handshake(node->openssl_ssl);
        if (ssl_chk == 1)
        {   
            chk = _abcdk_asio_openssl_verify_result(node);
            if(chk != 0)
                goto final_error;

            node->status = ABCDK_ASIO_STATUS_STABLE;
        }
        else
        {   
            /*必须通过返回值获取出错码。*/
            ssl_err = SSL_get_error(node->openssl_ssl, ssl_chk);

            if (ssl_err == SSL_ERROR_WANT_READ)
            {
                chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_INPUT, 0);
                if (chk == 0)
                    goto final;
            }
            else if (ssl_err == SSL_ERROR_WANT_WRITE)
            {
                chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_OUTPUT, 0);
                if (chk == 0)
                    goto final;
            }
            else
            {
                /*其它的全部当作出错处理。*/
                _abcdk_asio_openssl_dump_errmsg(node,ssl_err);
            }
            
            /*Error .*/
            goto final_error;
        }
#endif //HEADER_SSL_H
    }

final:

    /*OK or AGAIN.*/
    return;

final_error:

    /*修改超时，使用超时检测器关闭。*/
    abcdk_epollex_timeout(node->ctx->epollex, node->fd, 1);
}

void _abcdk_asio_perform(abcdk_asio_t *ctx,time_t timeout)
{
    int ret = 0;
    abcdk_asio_node_t *node = NULL;
    abcdk_epoll_event_t e = {0};
    int chk;

    memset(&e, 0, sizeof(abcdk_epoll_event_t));
    chk = abcdk_epollex_wait(ctx->epollex, &e, timeout);
    if (chk < 0)
        return;

    node = (abcdk_asio_node_t *)e.data.ptr;

    //fprintf(stderr,"fd(%d)=%u\n",node->fd,e.events);

    if (e.events & ABCDK_EPOLL_ERROR)
    {
        _abcdk_asio_event_cb(node, ABCDK_ASIO_EVENT_CLOSE,&ret);

        /*释放引用。*/
        abcdk_epollex_unref(ctx->epollex, node->fd, e.events);
        /*解除绑定关系。*/
        abcdk_epollex_detach(ctx->epollex, node->fd);
    }
    else
    {

        if (e.events & ABCDK_EPOLL_OUTPUT)
        {
            if (node->status != ABCDK_ASIO_STATUS_STABLE)
            {
                _abcdk_asio_handshake(node);
                if (node->status == ABCDK_ASIO_STATUS_STABLE)
                    _abcdk_asio_event_cb(node, ABCDK_ASIO_EVENT_CONNECT,&ret);
            }
            else
            {
                _abcdk_asio_event_cb(node, ABCDK_ASIO_EVENT_OUTPUT,&ret);
            }

            /*无论连接状态如何，写权利必须内部释放，不能开放给应用层。*/
            abcdk_epollex_mark(ctx->epollex, node->fd, 0, ABCDK_EPOLL_OUTPUT);
        }

        if (e.events & ABCDK_EPOLL_INPUT)
        {
            if (node->flag == ABCDK_ASIO_FLAG_LISTEN)
            {
                /*每次处理一个新连接。*/
                _abcdk_asio_accept(node);

                /*释放监听权利，并注册监听事件。*/
                abcdk_epollex_mark(ctx->epollex, node->fd, ABCDK_EPOLL_INPUT, ABCDK_EPOLL_INPUT);
            }
            else
            {
                if (node->status != ABCDK_ASIO_STATUS_STABLE)
                {
                    _abcdk_asio_handshake(node);
                    if (node->status == ABCDK_ASIO_STATUS_STABLE)
                        _abcdk_asio_event_cb(node, ABCDK_ASIO_EVENT_CONNECT,&ret);

                    /*释放读权利。*/
                    abcdk_epollex_mark(ctx->epollex, node->fd, 0, ABCDK_EPOLL_INPUT);
                }
                else
                {
                    _abcdk_asio_event_cb(node, ABCDK_ASIO_EVENT_INPUT,&ret);

                    /*在数据的传输过程中，读权利的释放由应用层决定，因此下面这句一定不要打开。*/
                    //abcdk_epollex_mark(ctx->epollex, node->fd, 0, ABCDK_EPOLL_INPUT);
                }
            }
        }

        /*释放引用计数。*/
        chk = abcdk_epollex_unref(ctx->epollex, node->fd, e.events);
        assert(chk == 0);
    }
}

void *_abcdk_asio_worker(void *args)
{
    abcdk_asio_t *ctx = (abcdk_asio_t *)args;

    /*每隔3秒检查一次，给退出检测留出时间。*/
    while (!abcdk_atomic_load(&ctx->exitflag))
        _abcdk_asio_perform(ctx, 3000);

    return NULL;
}

void abcdk_asio_stop(abcdk_asio_t **ctx)
{
    abcdk_asio_t *ctx_p = NULL;

    if(!ctx || !*ctx)
        return;

    /*复制。*/
    ctx_p = *ctx;
    *ctx = NULL;

    /*退出。*/
    abcdk_atomic_store(&ctx_p->exitflag, 1);

    /*回收线程资源。*/
    abcdk_thread_join(&ctx_p->worker);

    abcdk_epollex_free(&ctx_p->epollex);
    abcdk_heap_free(ctx_p);
}

abcdk_asio_t * abcdk_asio_start(int max,int cpu)
{
    abcdk_asio_t *ctx = NULL;
    long opm = sysconf(_SC_OPEN_MAX);
    int chk;

    ctx = abcdk_heap_alloc(sizeof(abcdk_asio_t));
    if(!ctx)
        return NULL;

    ctx->epollex = abcdk_epollex_alloc(_abcdk_asio_cleanup_cb, ctx);

    /*如果未指定最大连接数量，则使用文件句柄数量的一半。*/
    ctx->max = ((max > 0) ? max : abcdk_align(opm / 2, 1));
    ctx->exitflag = 0;

    /*创建工作线程。*/
    ctx->worker.handle = 0;
    ctx->worker.routine = _abcdk_asio_worker;
    ctx->worker.opaque = ctx;
    chk = abcdk_thread_create(&ctx->worker, 1);
    if (chk != 0)
        goto final_error;

    abcdk_thread_setaffinity(ctx->worker.handle,&cpu);

    return ctx;

final_error:
    
    abcdk_asio_stop(&ctx);

    return NULL;
}

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
static int _abcdk_asio_openssl_alpn_select_cb(SSL *ssl, const unsigned char **out, unsigned char *outlen,
                                              const unsigned char *in, unsigned int inlen, void *arg)
{
    abcdk_asio_node_t *node_p;
    const unsigned char *srv;
    unsigned int srvlen;

    node_p = (abcdk_asio_node_t *)arg;

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

static void _abcdk_asio_openssl_set_alpn(abcdk_asio_node_t *node)
{

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
    SSL_CTX_set_alpn_select_cb(node->openssl_ctx, _abcdk_asio_openssl_alpn_select_cb, (void *)node);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H
}

static int _abcdk_asio_ssl_init(abcdk_asio_node_t *node,int listen_flag)
{
    int ssl_chk;
    int ssl_err;
    int chk;

    if (node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI)
    {
#ifdef HEADER_SSL_H
        node->openssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(listen_flag,(node->cfg.pki_check_cert ? node->cfg.pki_ca_file : NULL),
                                                                   (node->cfg.pki_check_cert ? node->cfg.pki_ca_path : NULL),
                                                                   node->cfg.pki_cert_file, node->cfg.pki_key_file, NULL);

        if (!node->openssl_ctx)
        {
            abcdk_asio_trace_output(node,LOG_WARNING, "加载证书或私钥失败，无法创建SSL环境。");
            return -2;
        }
            
        /*设置密码套件。*/
        if(node->cfg.pki_cipher_list)
        {
            ssl_chk = SSL_CTX_set_cipher_list(node->openssl_ctx,node->cfg.pki_cipher_list);
            if(ssl_chk != 1)
            {
                ssl_err = SSL_get_error(node->openssl_ssl, ssl_chk);
                _abcdk_asio_openssl_dump_errmsg(node,ssl_err);
                return -3;
            }
        } 

        /*设置下层协议。*/
        if(node->cfg.pki_next_proto)
            _abcdk_asio_openssl_set_alpn(node);
#else 
        abcdk_asio_trace_output(node,LOG_WARNING, "构建时未包含相关组件，无法创建SSL环境。");
        return -22;
#endif // HEADER_SSL_H
    }
    else if (node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_ENIGMA)
    {
        node->easyssl_ssl = abcdk_easyssl_create_from_file(node->cfg.enigma_key_file,ABCDK_EASYSSL_SCHEME_ENIGMA,node->cfg.enigma_salt_size);
        if (!node->easyssl_ssl)
        {
            abcdk_asio_trace_output(node,LOG_WARNING, "加载共享钥失败，无法创建SSL环境。");
            return -2;
        }

        /*仅用于验证。*/
        abcdk_easyssl_destroy(&node->easyssl_ssl);
    }
    else if (node->cfg.ssl_scheme == ABCDK_ASIO_SSL_SCHEME_PKI_ON_ENIGMA)
    {
#ifdef HEADER_SSL_H
        node->openssl_bio = abcdk_BIO_s_easyssl(node->cfg.enigma_key_file,ABCDK_EASYSSL_SCHEME_ENIGMA,node->cfg.enigma_salt_size);
        if (!node->openssl_bio)
        {
            abcdk_asio_trace_output(node,LOG_WARNING, "加载共享钥失败，无法创建SSL环境。");
            return -2;
        }

        /*仅用于验证。*/
        abcdk_BIO_destroy(&node->openssl_bio);

        node->openssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(listen_flag,(node->cfg.pki_check_cert ? node->cfg.pki_ca_file : NULL),
                                                                   (node->cfg.pki_check_cert ? node->cfg.pki_ca_path : NULL),
                                                                   node->cfg.pki_cert_file, node->cfg.pki_key_file, NULL);

        if (!node->openssl_ctx)
        {
            abcdk_asio_trace_output(node,LOG_WARNING, "加载证书或私钥失败，无法创建SSL环境。");
            return -2;
        }
                    
        /*设置密码套件。*/
        if(node->cfg.pki_cipher_list)
        {
            ssl_chk = SSL_CTX_set_cipher_list(node->openssl_ctx,node->cfg.pki_cipher_list);
            if(ssl_chk != 1)
            {
                ssl_err = SSL_get_error(node->openssl_ssl, ssl_chk);
                _abcdk_asio_openssl_dump_errmsg(node,ssl_err);
                return -3;
            }
        } 

        /*设置下层协议。*/
        _abcdk_asio_openssl_set_alpn(node);

#else 
        abcdk_asio_trace_output(node,LOG_WARNING, "构建时未包含相关组件，无法创建SSL环境。");
        return -22;
#endif // HEADER_SSL_H
    }

    return 0;
}

int abcdk_asio_listen(abcdk_asio_node_t *node, abcdk_sockaddr_t *addr, abcdk_asio_config_t *cfg)
{
    abcdk_asio_node_t *node_p = NULL;
    epoll_data_t ep_data;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && addr != NULL && cfg != NULL);
    ABCDK_ASSERT(cfg->prepare_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");
    ABCDK_ASSERT(cfg->event_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");

    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_asio_refer(node);

    /*检测最大连接数量限制。*/
    if(abcdk_epollex_count(node_p->ctx->epollex) >= node_p->ctx->max)
        goto final_error;

    node_p->flag = ABCDK_ASIO_FLAG_LISTEN;
    node_p->status = ABCDK_ASIO_STATUS_STABLE;
    node_p->cfg = *cfg;

    /*修复不支持的配置。*/
    node_p->cfg.enigma_key_file = (node_p->cfg.enigma_key_file?node_p->cfg.enigma_key_file:"");
    node_p->cfg.enigma_salt_size = ABCDK_CLAMP(node_p->cfg.enigma_salt_size,0,256);

    /*UNIX需要特殊复制一下。*/
    if(addr->family == AF_UNIX)
    {
        node_p->local.family = AF_UNIX;
        strcpy(node_p->local.addr_un.sun_path,addr->addr_un.sun_path);
    }
    else
    {
        node_p->local = *addr;
    }
    
    node_p->fd = abcdk_socket(node_p->local.family, 0);
    if (node_p->fd < 0)
        goto final_error;

    /*端口复用，用于快速重启恢复。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(node_p->fd, SOL_SOCKET, SO_REUSEPORT, &sock_flag, 2);
    if (chk != 0)
        goto final_error;

    /*地址复用，用于快速重启恢复。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(node_p->fd, SOL_SOCKET, SO_REUSEADDR, &sock_flag, 2);
    if (chk != 0)
        goto final_error;

    if(addr->family == AF_INET6)
    {
        /*IPv6仅支持IPv6。*/
        sock_flag = 1;
        chk = abcdk_sockopt_option_int(node_p->fd, IPPROTO_IPV6, IPV6_V6ONLY, &sock_flag, 2);
        if (chk != 0)
            goto final_error;
    }

    chk = abcdk_bind(node_p->fd, &node_p->local);
    if (chk != 0) 
        goto final_error;

    chk = listen(node_p->fd, SOMAXCONN);
    if (chk != 0)
        goto final_error;
    
    chk = abcdk_fflag_add(node_p->fd,O_NONBLOCK);
    if(chk != 0)
        goto final_error;

    chk = _abcdk_asio_ssl_init(node,1);
    if(chk != 0)
        goto final_error;

    /*节点加入epoll池中。在解除绑定关系前，节点不会被释放。*/
    ep_data.ptr = node_p;
    chk = abcdk_epollex_attach(node_p->ctx->epollex,node_p->fd, &ep_data);
    if (chk != 0)
        goto final_error;
    
    /*关闭超时。*/
    abcdk_epollex_timeout(node_p->ctx->epollex, node_p->fd, 0);
    abcdk_epollex_mark(node_p->ctx->epollex, node_p->fd, ABCDK_EPOLL_INPUT, 0);

    return 0;

final_error:

    abcdk_asio_unref(&node_p);

    return -1;
}

int abcdk_asio_connect(abcdk_asio_node_t *node, abcdk_sockaddr_t *addr, abcdk_asio_config_t *cfg)
{
    abcdk_asio_node_t *node_p = NULL;
    epoll_data_t ep_data;
    socklen_t addr_len;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && addr != NULL && cfg != NULL);
    ABCDK_ASSERT(cfg->event_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");
    
    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_asio_refer(node);

    /*检测最大连接数量限制。*/
    if(abcdk_epollex_count(node_p->ctx->epollex) >= node_p->ctx->max)
        goto final_error;

    node_p->flag = ABCDK_ASIO_FLAG_CLIENT;
    node_p->status = ABCDK_ASIO_STATUS_SYNC;
    node_p->cfg = *cfg;

    /*修复不支持的配置。*/
    node_p->cfg.enigma_key_file = (node_p->cfg.enigma_key_file?node_p->cfg.enigma_key_file:"");
    node_p->cfg.enigma_salt_size = ABCDK_CLAMP(node_p->cfg.enigma_salt_size,0,256);
    
    addr_len = sizeof(abcdk_sockaddr_t);
    if(addr->family == AF_UNIX)
    {
        addr_len = SUN_LEN(&addr->addr_un);
        node_p->remote.family = AF_UNIX;
        strcpy(node_p->remote.addr_un.sun_path,addr->addr_un.sun_path);
    }
    else if(addr->family == AF_INET)
    {
        addr_len = sizeof(struct sockaddr_in);
        node_p->remote = *addr;
    }
    else if(addr->family == AF_INET6)
    {
        addr_len = sizeof(struct sockaddr_in6);
        node_p->remote = *addr;
    }

    node_p->fd = abcdk_socket(node_p->remote.family, 0);
    if (node_p->fd < 0)
        goto final_error;

    chk = abcdk_fflag_add(node_p->fd,O_NONBLOCK);
    if(chk != 0)
        goto final_error;

    chk = connect(node_p->fd, &node_p->remote.addr, addr_len);
    if(chk == 0)
        goto final;

    if (errno != EAGAIN && errno != EINPROGRESS)
        goto final_error;

    chk = _abcdk_asio_ssl_init(node,0);
    if(chk != 0)
        goto final_error;

final:

    /*节点加入epoll池中。在解除绑定关系前，节点不会被释放。*/
    ep_data.ptr = node_p;
    chk = abcdk_epollex_attach(node_p->ctx->epollex, node_p->fd, &ep_data);
    if (chk != 0)
        goto final_error;

    abcdk_epollex_timeout(node_p->ctx->epollex, node_p->fd, 180 * 1000);
    abcdk_epollex_mark(node_p->ctx->epollex, node_p->fd, ABCDK_EPOLL_OUTPUT, 0);

    return 0;

final_error:

    abcdk_asio_unref(&node_p);

    return -1;
}

int abcdk_asio_entrust(abcdk_asio_node_t *node, int fd, abcdk_asio_config_t *cfg)
{
    abcdk_asio_node_t *node_p = NULL;
    epoll_data_t ep_data;
    socklen_t addr_len;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && fd >= 0 && cfg != NULL);
    ABCDK_ASSERT(cfg->event_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");
    
    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_asio_refer(node);

    /*检测最大连接数量限制。*/
    if(abcdk_epollex_count(node_p->ctx->epollex) >= node_p->ctx->max)
        goto final_error;

    node_p->flag = ABCDK_ASIO_FLAG_CLIENT;
    node_p->status = ABCDK_ASIO_STATUS_SYNC;
    node_p->cfg = *cfg;

    /*修复不支持的配置。*/
    node_p->cfg.enigma_key_file = (node_p->cfg.enigma_key_file?node_p->cfg.enigma_key_file:"");
    node_p->cfg.enigma_salt_size = ABCDK_CLAMP(node_p->cfg.enigma_salt_size,0,256);

    /*绑定文件句柄。*/
    node_p->fd = fd;
     
    chk = abcdk_fflag_add(node_p->fd,O_NONBLOCK);
    if(chk != 0)
        goto final_error;

    chk = _abcdk_asio_ssl_init(node,0);
    if(chk != 0)
        goto final_error;

    /*节点加入epoll池中。在解除绑定关系前，节点不会被释放。*/
    ep_data.ptr = node_p;
    chk = abcdk_epollex_attach(node_p->ctx->epollex, node_p->fd, &ep_data);
    if (chk != 0)
        goto final_error;

    abcdk_epollex_timeout(node_p->ctx->epollex, node_p->fd, 180 * 1000);
    abcdk_epollex_mark(node_p->ctx->epollex, node_p->fd, ABCDK_EPOLL_OUTPUT, 0);

    return 0;

final_error:

    abcdk_asio_unref(&node_p);

    return -1;
}


void _abcdk_asio_input_hook(abcdk_asio_node_t *node)
{
    int ret = 0;
    ssize_t rlen = 0,pos = 0;
    size_t remain = 0;

    /*当未注册请求数据到达通知回调函数时，直接发事件通知。*/
    if(!node->cfg.request_cb)
    {
        node->cfg.event_cb(node,ABCDK_ASIO_EVENT_INPUT,&ret);
        return;
    }

NEXT_RECV:

    /*重置这些变量，非常重要。*/
    rlen = pos = 0;
    remain = 0;

    /*收。*/
    rlen = abcdk_asio_recv(node, node->in_buffer->pptrs[0], node->in_buffer->sizes[0]);
    if (rlen <= 0)
    {
        abcdk_asio_recv_watch(node);
        return;
    }

NEXT_REQ:

    node->cfg.request_cb(node, ABCDK_PTR2VPTR(node->in_buffer->pptrs[0], pos), rlen - pos, &remain);
    pos += (rlen - pos) - remain;

    if (pos < rlen)
        goto NEXT_REQ;
    else
        goto NEXT_RECV;//由于缓存里可能还有剩余数据，必须要清空缓存，才能重新进入监听状态。
}

void _abcdk_asio_output_hook(abcdk_asio_node_t *node)
{
    int ret = 0;
    abcdk_tree_t *p;
    ssize_t slen;
    int chk;

NEXT_MSG:

    /*从队列头部开始发送。*/
    abcdk_mutex_lock(node->out_locker,1);
    p = abcdk_tree_child(node->out_queue,1);
    abcdk_mutex_unlock(node->out_locker);

    /*通知应用层，发送队列空闲。*/
    if(!p)
    {
        node->cfg.event_cb(node,ABCDK_ASIO_EVENT_OUTPUT,&ret);
        return;
    }

    /*
     * 发。
     * 
     * 警告：重发数据时参数不能改变(指针和长度)。
    */
    slen = abcdk_asio_send(node, ABCDK_PTR2VPTR(p->obj->pptrs[0], node->out_pos), p->obj->sizes[0] - node->out_pos);
    if (slen <= 0)
    {
        abcdk_asio_send_watch(node);
        return;
    }

    /*滚动发送游标。*/
    node->out_pos += slen;

    /*当前节点未发送完整，则继续发送。*/
    if (node->out_pos < p->obj->sizes[0])
        goto NEXT_MSG;

    /*发送游标归零。*/
    node->out_pos = 0;

    /*从队列中删除已经发送完整的节点。*/
    abcdk_mutex_lock(node->out_locker,1);
    abcdk_tree_unlink(p);
    abcdk_tree_free(&p);
    abcdk_mutex_unlock(node->out_locker);

    /*并继续发送剩余节点。*/
    goto NEXT_MSG;
}

int abcdk_asio_post(abcdk_asio_node_t *node, abcdk_object_t *data)
{
    abcdk_tree_t *p;

    assert(node != NULL && data != NULL);
    assert(data->pptrs[0] != NULL && data->sizes[0] > 0);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if(!node->flag ||!node->status)
        return -3;

    if(node->flag == ABCDK_ASIO_FLAG_LISTEN)
        return -2;

    p = abcdk_tree_alloc(data);
    if(!p)
        return -1;

    abcdk_mutex_lock(node->out_locker,1);
    abcdk_tree_insert2(node->out_queue,p,0);
    abcdk_mutex_unlock(node->out_locker);

    if(node->status == ABCDK_ASIO_STATUS_STABLE)
        abcdk_asio_send_watch(node);

    return 0;
}

int abcdk_asio_post_buffer(abcdk_asio_node_t *node, const void *data,size_t size)
{
    abcdk_object_t *obj;
    int chk;

    assert(node != NULL && data != NULL && size >0);

    obj = abcdk_object_copyfrom(data,size);
    if(!obj)
        return -1;

    chk = abcdk_asio_post(node,obj);
    if(chk == 0)
        return 0;

    /*删除投递失败的。*/
    abcdk_object_unref(&obj);
    return chk;
}

int abcdk_asio_post_vformat(abcdk_asio_node_t *node, int max, const char *fmt, va_list ap)
{
    abcdk_object_t *obj;
    int chk;

    assert(node != NULL && max > 0 && fmt != NULL);

    obj = abcdk_object_vprintf(max,fmt,ap);
    if(!obj)
        return -1;
    
    chk = abcdk_asio_post(node,obj);
    if(chk == 0)
        return 0;

    /*删除投递失败的。*/
    abcdk_object_unref(&obj);
    return chk;
}

int abcdk_asio_post_format(abcdk_asio_node_t *node, int max, const char *fmt, ...)
{
    int chk;

    assert(node != NULL && max > 0 && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    chk = abcdk_asio_post_vformat(node, max, fmt, ap);
    va_end(ap);

    return chk;
}