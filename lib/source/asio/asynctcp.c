/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/asio/asynctcp.h"

/** 简单的异步TCP通讯。 */
struct _abcdk_asynctcp
{
    /** epollex 环境。*/
    abcdk_epollex_t *epollex;

    /** 工作线程。*/
    abcdk_thread_t worker;

    /** 最大连接数量。*/
    int max;

    /** 退出标志。0： 运行，!0：停止。*/
    volatile int exitflag;

};// abcdk_asynctcp_t;

/** 异步TCP节点。 */
struct _abcdk_asynctcp_node
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_ASYNCTCP_NODE_MAGIC 123456789

    /** 引用计数器。*/
    volatile int refcount;

    /** 
     * 通讯环境指针。
     * 
     * @note 仅复制。
    */
    abcdk_asynctcp_t *ctx;

    /** 标识句柄来源。*/
    volatile int flag;
#define ABCDK_ASYNCTCP_FLAG_CLIENT   1
#define ABCDK_ASYNCTCP_FLAG_LISTEN   2
#define ABCDK_ASYNCTCP_FLAG_ACCPET   3

    /** 标识当前句柄状态。*/
    volatile int status;
#define ABCDK_ASYNCTCP_STATUS_STABLE        1
#define ABCDK_ASYNCTCP_STATUS_SYNC          2
#define ABCDK_ASYNCTCP_STATUS_OPENSSL_SYNC  3
#define ABCDK_ASYNCTCP_STATUS_EASYSSL_SYNC  4

    /** 本机地址。*/
    abcdk_sockaddr_t local;

    /** 远端地址。*/
    abcdk_sockaddr_t remote;

    /** 句柄。*/
    int fd;

    /*安全方案*/
    int ssl_scheme;
#define ABCDK_ASYNCTCP_SSL_SCHEME_RAW       0
#define ABCDK_ASYNCTCP_SSL_SCHEME_OPENSSL   1
#define ABCDK_ASYNCTCP_SSL_SCHEME_EASYSSL   2

    /** openssl环境指针。*/
    SSL *openssl_ctx;

    /** easyssl环境指针。*/
    abcdk_easyssl_t *easyssl_ctx;

    /** 工作线程。*/
    volatile pthread_t worker;

    /** 回调函数。*/
    abcdk_asynctcp_callback_t *callback;
    abcdk_asynctcp_callback_t cb_cp;

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
    struct _abcdk_asynctcp_node *from_listen;

};// abcdk_asynctcp_node_t;

void abcdk_asynctcp_unref(abcdk_asynctcp_node_t **node)
{
    abcdk_asynctcp_node_t *node_p = NULL;

    if (!node || !*node)
        return;

    node_p = *node;
    *node = NULL;

    assert(node_p->magic == ABCDK_ASYNCTCP_NODE_MAGIC);

    if (abcdk_atomic_fetch_and_add(&node_p->refcount, -1) != 1)
        return;

    assert(node_p->refcount == 0);

    node_p->magic = 0xcccccccc;

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_free(&node_p->openssl_ctx);
#endif //HEADER_SSL_H

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
    abcdk_asynctcp_unref(&node_p->from_listen);
    abcdk_heap_free(node_p);
}

abcdk_asynctcp_node_t *abcdk_asynctcp_refer(abcdk_asynctcp_node_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_asynctcp_node_t *abcdk_asynctcp_alloc(abcdk_asynctcp_t *ctx,size_t userdata, void (*free_cb)(void *userdata))
{
    abcdk_asynctcp_node_t *node = NULL;

    assert(ctx != NULL);

    node = (abcdk_asynctcp_node_t *)abcdk_heap_alloc(sizeof(abcdk_asynctcp_node_t));
    if(!node)
        return NULL;

    node->magic = ABCDK_ASYNCTCP_NODE_MAGIC;
    node->refcount = 1;
    node->ctx = ctx;
    node->fd = -1;
    node->userdata = abcdk_object_alloc3(userdata,1);
    node->userdata_free_cb = free_cb;
    node->out_queue = abcdk_tree_alloc3(1);
    node->out_locker = abcdk_mutex_create();
    node->out_pos = 0;
    node->in_buffer = NULL;
    node->from_listen = NULL;
    node->ssl_scheme = ABCDK_ASYNCTCP_SSL_SCHEME_RAW;
    node->openssl_ctx = NULL;
    node->easyssl_ctx = NULL;

    return node;
}

int abcdk_asynctcp_upgrade2openssl(abcdk_asynctcp_node_t *node,SSL_CTX *ssl_ctx)
{
    assert(node != NULL && ssl_ctx != NULL);

    ABCDK_ASSERT(node->openssl_ctx == NULL,"仅可以升级一次。");

#ifdef HEADER_SSL_H
    node->openssl_ctx = abcdk_openssl_ssl_alloc(ssl_ctx);
#else   
    abcdk_trace_output(LOG_WARNING, "构建时未包含SSL组件，无法创建SSL环境。");
#endif //HEADER_SSL_H
    if(!node->openssl_ctx)
        return -1;

    node->ssl_scheme = ABCDK_ASYNCTCP_SSL_SCHEME_OPENSSL;

    return 0;

}

int abcdk_asynctcp_upgrade2easyssl(abcdk_asynctcp_node_t *node,abcdk_easyssl_t *ssl_ctx)
{
    assert(node != NULL && ssl_ctx != NULL);

    ABCDK_ASSERT(node->easyssl_ctx == NULL,"仅可以升级一次。");

    node->easyssl_ctx = ssl_ctx;

    node->ssl_scheme = ABCDK_ASYNCTCP_SSL_SCHEME_EASYSSL;

    return 0;
}

SSL *abcdk_asynctcp_openssl_ctx(abcdk_asynctcp_node_t *node)
{
    assert(node != NULL);

    if(node->openssl_ctx)
        return node->openssl_ctx;

    return NULL;
}

void *abcdk_asynctcp_get_userdata(abcdk_asynctcp_node_t *node)
{
    assert(node != NULL);

    return node->userdata->pptrs[0];
}

void *abcdk_asynctcp_set_userdata(abcdk_asynctcp_node_t *node,void *opaque)
{
    void *old;

    assert(node != NULL);

    old = node->userdata->pptrs[0];
    node->userdata->pptrs[0] = (uint8_t*)opaque;

    return old;
}

int abcdk_asynctcp_set_timeout(abcdk_asynctcp_node_t *node, time_t timeout)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    chk = abcdk_epollex_timeout(node->ctx->epollex, node->fd, timeout);

    return chk;
}

int abcdk_asynctcp_get_sockaddr(abcdk_asynctcp_node_t *node, abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote)
{
    assert(node != NULL);

    if (local && node->local.family)
        abcdk_sockaddr_copy(&node->local,local);

    if (remote && node->remote.family)
        abcdk_sockaddr_copy(&node->remote,remote);

    return 0;
}

int abcdk_asynctcp_get_sockaddr_str(abcdk_asynctcp_node_t *node, char local[NAME_MAX],char remote[NAME_MAX])
{
    assert(node != NULL);

    if(local && node->local.family)
        abcdk_sockaddr_to_string(local,&node->local);

    if(remote && node->remote.family)
        abcdk_sockaddr_to_string(remote,&node->remote);

    return 0;
}

ssize_t abcdk_asynctcp_recv(abcdk_asynctcp_node_t *node, void *buf, size_t size)
{
    ssize_t rsize = 0,rsize_all = 0;
    int chk;

    assert(node != NULL && buf != NULL && size >0);

    /*仅消息循环线程拥有读权利。*/
    chk = abcdk_thread_leader_test(&node->worker);
    ABCDK_ASSERT(chk == 0,"当前线程没有读权利。");

    while (rsize_all < size)
    {
        if(node->ssl_scheme == ABCDK_ASYNCTCP_SSL_SCHEME_OPENSSL)
            rsize = SSL_read(node->openssl_ctx,ABCDK_PTR2PTR(void,buf,rsize_all),size-rsize_all);
        else if(node->ssl_scheme == ABCDK_ASYNCTCP_SSL_SCHEME_EASYSSL)
            rsize = abcdk_easyssl_read(node->easyssl_ctx,ABCDK_PTR2PTR(void,buf,rsize_all),size-rsize_all);
        else 
            rsize = read(node->fd,ABCDK_PTR2PTR(void,buf,rsize_all),size-rsize_all);
        
        if(rsize <=0)
            break;
        
        rsize_all += rsize;
    }

    return rsize_all;
}

int abcdk_asynctcp_recv_watch(abcdk_asynctcp_node_t *node)
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

ssize_t abcdk_asynctcp_send(abcdk_asynctcp_node_t *node, void *buf, size_t size)
{
    ssize_t wsize = 0,wsize_all = 0;
    int chk;

    assert(node != NULL && buf != NULL && size > 0);

    /*仅消息循环线程拥有写权利。*/
    chk = abcdk_thread_leader_test(&node->worker);
    ABCDK_ASSERT(chk == 0,"当前线程没有写权利。");

    while (wsize_all < size)
    {
        if(node->ssl_scheme == ABCDK_ASYNCTCP_SSL_SCHEME_OPENSSL)
            wsize = SSL_write(node->openssl_ctx,ABCDK_PTR2PTR(void,buf,wsize_all),size-wsize_all);
        else if(node->ssl_scheme == ABCDK_ASYNCTCP_SSL_SCHEME_EASYSSL)
            wsize = abcdk_easyssl_write(node->easyssl_ctx,ABCDK_PTR2PTR(void,buf,wsize_all),size-wsize_all);
        else 
            wsize = write(node->fd,ABCDK_PTR2PTR(void,buf,wsize_all),size-wsize_all);
        
        if(wsize <=0)
            break;
        
        wsize_all += wsize;
    }

    return wsize_all;
}

int abcdk_asynctcp_send_watch(abcdk_asynctcp_node_t *node)
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

void _abcdk_asynctcp_cleanup_cb(epoll_data_t *data, void *opaque)
{
    abcdk_asynctcp_t *ctx = (abcdk_asynctcp_t *)opaque;
    abcdk_asynctcp_node_t *node = NULL;

    node = (abcdk_asynctcp_node_t *)data->ptr;
    abcdk_asynctcp_unref(&node);
}

void _abcdk_asynctcp_prepare_cb(abcdk_asynctcp_node_t **node, abcdk_asynctcp_node_t *listen)
{
    /*通知应用层处理事件。*/
    if (listen->callback->prepare_cb)
        listen->callback->prepare_cb(node, listen);
}

/*声明输入事件钩子函数。*/
void _abcdk_asynctcp_input_hook(abcdk_asynctcp_node_t *node);

/*声明输出事件钩子函数。*/
void _abcdk_asynctcp_output_hook(abcdk_asynctcp_node_t *node);

void _abcdk_asynctcp_event_cb(abcdk_asynctcp_node_t *node,uint32_t event, int *result)
{
    /*初次建立连接，取消默认的发和收超时设置。*/



    /*绑定工作线程。*/
    abcdk_thread_leader_vote(&node->worker);

    /*通知应用层处理事件。*/
    if(event == ABCDK_ASYNCTCP_EVENT_INPUT)
        _abcdk_asynctcp_input_hook(node);
    else if(event == ABCDK_ASYNCTCP_EVENT_OUTPUT)
        _abcdk_asynctcp_output_hook(node);
    else 
        node->callback->event_cb(node, event, result);

    /*解绑工作线程。*/
    abcdk_thread_leader_quit(&node->worker);
}

void _abcdk_asynctcp_accept(abcdk_asynctcp_node_t *listen)
{
    abcdk_asynctcp_node_t *node = NULL;
    epoll_data_t ep_data;
    int chk;

    /*通知初始化。*/
    _abcdk_asynctcp_prepare_cb(&node, listen);
    if (!node)
        return;

    /*配置参数。*/
    node->flag = ABCDK_ASYNCTCP_FLAG_ACCPET;
    node->status = ABCDK_ASYNCTCP_STATUS_SYNC;

    /*记住来源。*/
    node->from_listen = abcdk_asynctcp_refer(listen);

    /*复制通讯环境指针。*/
    node->ctx = listen->ctx;
    /*复制监听环境的回调函数指针。*/
    node->callback = listen->callback;

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
    _abcdk_asynctcp_event_cb(node,ABCDK_ASYNCTCP_EVENT_ACCEPT,&chk);
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
    _abcdk_asynctcp_event_cb(node, ABCDK_ASYNCTCP_EVENT_INTERRUPT,&chk);
    abcdk_asynctcp_unref(&node);
    
    return;
}

void _abcdk_asynctcp_handshake(abcdk_asynctcp_node_t *node)
{
    socklen_t sock_len = 0;
    int sock_flag = 1;
    int ssl_chk;
    int ssl_err;
    struct timeval tv;
    int chk;

    if (node->status == ABCDK_ASYNCTCP_STATUS_SYNC)
    {
        chk = abcdk_poll(node->fd, 0x02, 0);
        if (chk > 0)
        {
            if(node->ssl_scheme == ABCDK_ASYNCTCP_SSL_SCHEME_OPENSSL)
                node->status = ABCDK_ASYNCTCP_STATUS_OPENSSL_SYNC;
            else if(node->ssl_scheme == ABCDK_ASYNCTCP_SSL_SCHEME_EASYSSL)
                node->status = ABCDK_ASYNCTCP_STATUS_EASYSSL_SYNC;
            else 
                node->status = ABCDK_ASYNCTCP_STATUS_STABLE;
        }
        else
        {
            chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_OUTPUT, 0);
            if (chk != 0)
                goto final_error;
            else 
                goto final;
        }

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
        abcdk_sockopt_option_timeout(node->fd,SO_RCVTIMEO,&tv,2);
        abcdk_sockopt_option_timeout(node->fd,SO_SNDTIMEO,&tv,2);

        /*修改保活参数，以防在远程断电的情况下本地无法检测到连接断开信号。*/

        /*开启keepalive属性*/
        sock_flag = 1;
        abcdk_sockopt_option_int(node->fd,SOL_SOCKET, SO_KEEPALIVE,&sock_flag,2);

        /*连接在60秒内没有任何数据往来，则进行探测。*/
        sock_flag = 60;
        abcdk_sockopt_option_int(node->fd,IPPROTO_TCP, TCP_KEEPIDLE,&sock_flag,2);

        /*探测时发包的时间间隔为5秒。*/
        sock_flag = 5;
        abcdk_sockopt_option_int(node->fd,IPPROTO_TCP, TCP_KEEPINTVL,&sock_flag,2);

        /*探测尝试的次数.如果第一次探测包就收到响应，则后两次的不再发。*/
        sock_flag = 3;
        abcdk_sockopt_option_int(node->fd,IPPROTO_TCP, TCP_KEEPCNT,&sock_flag,2);

        /*关闭延迟发送。*/
        sock_flag = 1;
        abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_NODELAY,&sock_flag, 2);
    }

#ifdef HEADER_SSL_H      
    if (node->status == ABCDK_ASYNCTCP_STATUS_OPENSSL_SYNC)
    {
        if (SSL_get_fd(node->openssl_ctx) != node->fd)
        {
            SSL_set_fd(node->openssl_ctx, node->fd);
            
            if (node->flag == ABCDK_ASYNCTCP_FLAG_ACCPET)
                SSL_set_accept_state(node->openssl_ctx);
            else if (node->flag == ABCDK_ASYNCTCP_FLAG_CLIENT)
                SSL_set_connect_state(node->openssl_ctx);
            else
                goto final_error;

#ifdef SSL_OP_NO_RENEGOTIATION
            SSL_set_options(node->openssl_ctx, SSL_OP_NO_RENEGOTIATION);
#endif
        }

        ssl_chk = SSL_do_handshake(node->openssl_ctx);
        if (ssl_chk == 1)
        {   
            node->status = ABCDK_ASYNCTCP_STATUS_STABLE;
        }
        else
        {   
            /*必须通过返回值获取出错码。*/
            ssl_err = SSL_get_error(node->openssl_ctx, ssl_chk);

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
                ERR_print_errors_fp(stderr);
            }
            
            /*Error .*/
            goto final_error;
        }
    }
    else
#endif //HEADER_SSL_H
    if (node->status == ABCDK_ASYNCTCP_STATUS_EASYSSL_SYNC)
    {
        abcdk_easyssl_set_fd(node->easyssl_ctx, node->fd,0);
        abcdk_easyssl_set_fd(node->easyssl_ctx, node->fd,1);
        node->status = ABCDK_ASYNCTCP_STATUS_STABLE;
    }

final:

    /*OK or AGAIN.*/
    return;

final_error:

    /*修改超时，使用超时检测器关闭。*/
    abcdk_epollex_timeout(node->ctx->epollex, node->fd, 1);
}

void _abcdk_asynctcp_perform(abcdk_asynctcp_t *ctx,time_t timeout)
{
    int ret = 0;
    abcdk_asynctcp_node_t *node = NULL;
    abcdk_epoll_event_t e = {0};
    int chk;

    memset(&e, 0, sizeof(abcdk_epoll_event_t));
    chk = abcdk_epollex_wait(ctx->epollex, &e, timeout);
    if (chk < 0)
        return;

    node = (abcdk_asynctcp_node_t *)e.data.ptr;

    //fprintf(stderr,"fd(%d)=%u\n",node->fd,e.events);

    if (e.events & ABCDK_EPOLL_ERROR)
    {
        _abcdk_asynctcp_event_cb(node, ABCDK_ASYNCTCP_EVENT_CLOSE,&ret);

        /*释放引用。*/
        abcdk_epollex_unref(ctx->epollex, node->fd, e.events);
        /*解除绑定关系。*/
        abcdk_epollex_detach(ctx->epollex, node->fd);
    }
    else
    {

        if (e.events & ABCDK_EPOLL_OUTPUT)
        {
            if (node->status != ABCDK_ASYNCTCP_STATUS_STABLE)
            {
                _abcdk_asynctcp_handshake(node);
                if (node->status == ABCDK_ASYNCTCP_STATUS_STABLE)
                    _abcdk_asynctcp_event_cb(node, ABCDK_ASYNCTCP_EVENT_CONNECT,&ret);
            }
            else
            {
                _abcdk_asynctcp_event_cb(node, ABCDK_ASYNCTCP_EVENT_OUTPUT,&ret);
            }

            /*无论连接状态如何，写权利必须内部释放，不能开放给应用层。*/
            abcdk_epollex_mark(ctx->epollex, node->fd, 0, ABCDK_EPOLL_OUTPUT);
        }

        if (e.events & ABCDK_EPOLL_INPUT)
        {
            if (node->flag == ABCDK_ASYNCTCP_FLAG_LISTEN)
            {
                /*每次处理一个新连接。*/
                _abcdk_asynctcp_accept(node);

                /*释放监听权利，并注册监听事件。*/
                abcdk_epollex_mark(ctx->epollex, node->fd, ABCDK_EPOLL_INPUT, ABCDK_EPOLL_INPUT);
            }
            else
            {
                if (node->status != ABCDK_ASYNCTCP_STATUS_STABLE)
                {
                    _abcdk_asynctcp_handshake(node);
                    if (node->status == ABCDK_ASYNCTCP_STATUS_STABLE)
                        _abcdk_asynctcp_event_cb(node, ABCDK_ASYNCTCP_EVENT_CONNECT,&ret);

                    /*释放读权利。*/
                    abcdk_epollex_mark(ctx->epollex, node->fd, 0, ABCDK_EPOLL_INPUT);
                }
                else
                {
                    _abcdk_asynctcp_event_cb(node, ABCDK_ASYNCTCP_EVENT_INPUT,&ret);

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

void *_abcdk_asynctcp_worker(void *args)
{
    abcdk_asynctcp_t *ctx = (abcdk_asynctcp_t *)args;

    /*每隔3秒检查一次，给退出检测留出时间。*/
    while (!abcdk_atomic_load(&ctx->exitflag))
        _abcdk_asynctcp_perform(ctx, 3000);

    return NULL;
}

void abcdk_asynctcp_stop(abcdk_asynctcp_t **ctx)
{
    abcdk_asynctcp_t *ctx_p = NULL;

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

abcdk_asynctcp_t * abcdk_asynctcp_start(int max,int cpu)
{
    abcdk_asynctcp_t *ctx = NULL;
    long opm = sysconf(_SC_OPEN_MAX);
    int chk;

    ctx = abcdk_heap_alloc(sizeof(abcdk_asynctcp_t));
    if(!ctx)
        return NULL;

    ctx->epollex = abcdk_epollex_alloc(_abcdk_asynctcp_cleanup_cb, ctx);

    /*如果未指定最大连接数量，则使用文件句柄数量的一半。*/
    ctx->max = ((max > 0) ? max : abcdk_align(opm / 2, 1));
    ctx->exitflag = 0;

    /*创建工作线程。*/
    ctx->worker.handle = 0;
    ctx->worker.routine = _abcdk_asynctcp_worker;
    ctx->worker.opaque = ctx;
    chk = abcdk_thread_create(&ctx->worker, 1);
    if (chk != 0)
        goto final_error;

    abcdk_thread_setaffinity(ctx->worker.handle,&cpu);

    return ctx;

final_error:
    
    abcdk_asynctcp_stop(&ctx);

    return NULL;
}

int abcdk_asynctcp_listen(abcdk_asynctcp_node_t *node, abcdk_sockaddr_t *addr, abcdk_asynctcp_callback_t *cb)
{
    abcdk_asynctcp_node_t *node_p = NULL;
    epoll_data_t ep_data;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->prepare_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");
    ABCDK_ASSERT(cb->event_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");

    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_asynctcp_refer(node);

    /*检测最大连接数量限制。*/
    if(abcdk_epollex_count(node_p->ctx->epollex) >= node_p->ctx->max)
        goto final_error;

    node_p->flag = ABCDK_ASYNCTCP_FLAG_LISTEN;
    node_p->status = ABCDK_ASYNCTCP_STATUS_STABLE;
    node_p->cb_cp = *cb;
    node_p->callback = &node_p->cb_cp;

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
    if(chk != 0 )
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

    abcdk_asynctcp_unref(&node_p);

    return -1;
}

int abcdk_asynctcp_connect(abcdk_asynctcp_node_t *node, abcdk_sockaddr_t *addr, abcdk_asynctcp_callback_t *cb)
{
    abcdk_asynctcp_node_t *node_p = NULL;
    epoll_data_t ep_data;
    socklen_t addr_len;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->event_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");
    
    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_asynctcp_refer(node);

    /*检测最大连接数量限制。*/
    if(abcdk_epollex_count(node_p->ctx->epollex) >= node_p->ctx->max)
        goto final_error;

    node_p->flag = ABCDK_ASYNCTCP_FLAG_CLIENT;
    node_p->status = ABCDK_ASYNCTCP_STATUS_SYNC;
    node_p->cb_cp = *cb;
    node_p->callback = &node_p->cb_cp;
    
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
    if(chk != 0 )
        goto final_error;

    chk = connect(node_p->fd, &node_p->remote.addr, addr_len);
    if(chk == 0)
        goto final;

    if (errno != EAGAIN && errno != EINPROGRESS)
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

    abcdk_asynctcp_unref(&node_p);

    return -1;
}

void _abcdk_asynctcp_input_hook(abcdk_asynctcp_node_t *node)
{
    int ret = 0;
    ssize_t rlen = 0,pos = 0;
    size_t remain = 0;

    /*当未注册请求数据到达通知回调函数时，直接发事件通知。*/
    if(!node->callback->request_cb)
    {
        node->callback->event_cb(node,ABCDK_ASYNCTCP_EVENT_INPUT,&ret);
        return;
    }

    /*准备接收数据的缓存。*/
    if (!node->in_buffer)
    {
        node->in_buffer = abcdk_object_alloc2(256*1024);
        if (!node->in_buffer)
        {
            abcdk_asynctcp_set_timeout(node, 1);
            return;
        }
    }

    /*收。*/
    rlen = abcdk_asynctcp_recv(node, node->in_buffer->pptrs[0], node->in_buffer->sizes[0]);
    if (rlen <= 0)
    {
        abcdk_asynctcp_recv_watch(node);
        return;
    }

NEXT_REQ:

    node->callback->request_cb(node, ABCDK_PTR2VPTR(node->in_buffer->pptrs[0], pos), rlen - pos, &remain);
    pos += (rlen - pos) - remain;

    if (pos < rlen)
        goto NEXT_REQ;
    else
        abcdk_asynctcp_recv_watch(node);
}

void _abcdk_asynctcp_output_hook(abcdk_asynctcp_node_t *node)
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
        node->callback->event_cb(node,ABCDK_ASYNCTCP_EVENT_OUTPUT,&ret);
        return;
    }

    /*
     * 发。
     * 
     * 警告：重发数据时参数不能改变(指针和长度)。
    */
    slen = abcdk_asynctcp_send(node, ABCDK_PTR2VPTR(p->obj->pptrs[0], node->out_pos), p->obj->sizes[0] - node->out_pos);
    if (slen <= 0)
    {
        abcdk_asynctcp_send_watch(node);
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

int abcdk_asynctcp_post(abcdk_asynctcp_node_t *node, abcdk_object_t *data)
{
    abcdk_tree_t *p;

    assert(node != NULL && data != NULL);
    assert(data->pptrs[0] != NULL && data->sizes[0] > 0);

    /*没有确定节点属性和状态前，不能调用此接口。*/
    if(!node->flag ||!node->status)
        return -3;

    if(node->flag == ABCDK_ASYNCTCP_FLAG_LISTEN)
        return -2;

    p = abcdk_tree_alloc(data);
    if(!p)
        return -1;

    abcdk_mutex_lock(node->out_locker,1);
    abcdk_tree_insert2(node->out_queue,p,0);
    abcdk_mutex_unlock(node->out_locker);

    if(node->status == ABCDK_ASYNCTCP_STATUS_STABLE)
        abcdk_asynctcp_send_watch(node);

    return 0;
}

int abcdk_asynctcp_post_buffer(abcdk_asynctcp_node_t *node, const void *data,size_t size)
{
    abcdk_object_t *obj;
    int chk;

    assert(node != NULL && data != NULL && size >0);

    obj = abcdk_object_copyfrom(data,size);
    if(!obj)
        return -1;

    chk = abcdk_asynctcp_post(node,obj);
    if(chk == 0)
        return 0;

    /*删除投递失败的。*/
    abcdk_object_unref(&obj);
    return chk;
}

int abcdk_asynctcp_post_vformat(abcdk_asynctcp_node_t *node, int max, const char *fmt, va_list ap)
{
    abcdk_object_t *obj;
    int chk;

    assert(node != NULL && max > 0 && fmt != NULL);

    obj = abcdk_object_vprintf(max,fmt,ap);
    if(!obj)
        return -1;
    
    chk = abcdk_asynctcp_post(node,obj);
    if(chk == 0)
        return 0;

    /*删除投递失败的。*/
    abcdk_object_unref(&obj);
    return chk;
}

int abcdk_asynctcp_post_format(abcdk_asynctcp_node_t *node, int max, const char *fmt, ...)
{
    int chk;

    assert(node != NULL && max > 0 && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    chk = abcdk_asynctcp_post_vformat(node, max, fmt, ap);
    va_end(ap);

    return chk;
}