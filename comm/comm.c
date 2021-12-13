/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-comm/comm.h"

/** 通信环境。*/
typedef struct _abcdk_comm_ctx
{
    /** epollex 环境。*/
    abcdk_epollex_t *epollex_ctx;

    /** 事件循环标志。0 持续运行，!0 中止运行。*/
    volatile int loop_abort;

} abcdk_comm_ctx;

/** 节点信息。*/
typedef struct _abcdk_comm_node
{
    /** 标识句柄来源。*/
    int flag;
#define ABCDK_COMM_FLAG_CLIENT   1
#define ABCDK_COMM_FLAG_LISTEN   2
#define ABCDK_COMM_FLAG_ACCPET   3

    /** 标识当前句柄状态。*/
    int status;
#define ABCDK_COMM_STATUS_SYNC       1
#define ABCDK_COMM_STATUS_SSL_SYNC   2
#define ABCDK_COMM_STATUS_STABLE     3


    /** 本机地址。*/
    abcdk_sockaddr_t local;

    /** 远端地址。*/
    abcdk_sockaddr_t remote;

    /** 句柄。*/
    int fd;

    /** SSL/COMM环境指针。*/
#ifdef HEADER_SSL_H
    SSL_CTX *ssl_ctx;
    SSL *ssl;
#endif //HEADER_SSL_H

    /** Input事件读权利拥有者。*/
    volatile pthread_t input_user;

    /*
     * 应用层环境指针。
     *
     * 1、服务端指针会被新句柄复制使用。
     * 2、客户端指针仅会被当前句柄使用。
    */
    void *opaque;

} abcdk_comm_node;

void _abcdk_comm_node_free(abcdk_comm_node **node)
{
    abcdk_comm_node *node_p = NULL;
    if (!node)
        return;

    node_p = *node;

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_free(&node_p->ssl);
#endif //HEADER_SSL_H

    /*直接关闭，快速回收资源，不会处于time_wait状态。*/
    struct linger l = {1,0};
    abcdk_socket_option_linger(node_p->fd,&l,2);
    abcdk_closep(&node_p->fd);

    /*free and set NULL(0).*/
    abcdk_heap_free2((void **)node);
}

abcdk_comm_node *_abcdk_comm_node_alloc()
{
    return (abcdk_comm_node *)abcdk_heap_alloc(sizeof(abcdk_comm_node));
}

void _abcdk_comm_ctx_cleanup_cb(epoll_data_t *data, void *opaque)
{
    abcdk_comm_ctx *ctx = (abcdk_comm_ctx *)opaque;
    abcdk_comm_node *node = NULL;

    node = (abcdk_comm_node *)data->ptr;
    _abcdk_comm_node_free(&node);
}

int _abcdk_comm_ctx_init(void *opaque)
{
    abcdk_comm_ctx *ctx = (abcdk_comm_ctx *)opaque;

    ctx->epollex_ctx = abcdk_epollex_alloc(_abcdk_comm_ctx_cleanup_cb,ctx);
    ctx->loop_abort = 0;

    return 0;
}

abcdk_comm_ctx *_abcdk_comm_get_ctx()
{
    static volatile int init = 0;
    static abcdk_comm_ctx ctx = {0};
    int chk;

    chk = abcdk_once(&init, _abcdk_comm_ctx_init, &ctx);
    assert(chk >= 0);

    return &ctx;
}

abcdk_comm_node *_abcdk_comm_accept(abcdk_comm_node *node)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    abcdk_comm_node *node_sub = NULL;
    epoll_data_t ep_data;
    int chk;

    node_sub = _abcdk_comm_node_alloc();
    if (!node_sub)
        return NULL;
    
    node_sub->flag = ABCDK_COMM_FLAG_ACCPET;
    node_sub->status = ABCDK_COMM_STATUS_SYNC;

#ifdef HEADER_SSL_H    
    if(node->ssl_ctx)
    {
        node_sub->ssl = abcdk_openssl_ssl_alloc(node->ssl_ctx);
        if(!node_sub->ssl)
            goto final_error;
    }
#endif //HEADER_SSL_H

    node_sub->fd = abcdk_accept(node->fd, &node_sub->remote);
    if (node_sub->fd < 0)
        goto final_error;

    ep_data.ptr = node_sub;
    chk = abcdk_epollex_attach(comm_ctx->epollex_ctx, node_sub->fd, &ep_data);
    if(chk != 0)
        goto final_error;

    abcdk_epollex_timeout(comm_ctx->epollex_ctx, node_sub->fd, 30*1000);

    return node_sub;

final_error:

    _abcdk_comm_node_free(&node_sub);
    
    return NULL;
}

void abcdk_comm_handshake(abcdk_comm_node *node)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    int sock_flag = 1;
    int ssl_chk;
    int ssl_err;
    int chk;

    if (node->status == ABCDK_COMM_STATUS_SYNC)
    {
        chk = abcdk_poll(node->fd, 0x02, 0);
        if (chk > 0)
        {
#ifdef HEADER_SSL_H    
            if(node->ssl)
                node->status = ABCDK_COMM_STATUS_SSL_SYNC;
            else 
#endif //HEADER_SSL_H
                node->status = ABCDK_COMM_STATUS_STABLE;
        }
        else
        {
            chk = abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_OUTPUT, 0);
            if (chk != 0)
                goto final_error;
        }
    }

#ifdef HEADER_SSL_H      
    if (node->status == ABCDK_COMM_STATUS_SSL_SYNC)
    {
        if (node->flag == ABCDK_COMM_FLAG_ACCPET)
        {
            if (SSL_get_fd(node->ssl) != node->fd)
            {
                SSL_set_fd(node->ssl, node->fd);
                SSL_set_accept_state(node->ssl);
            }
        }
        else if (node->flag == ABCDK_COMM_FLAG_CLIENT)
        {
            if (SSL_get_fd(node->ssl) != node->fd)
            {
                SSL_set_fd(node->ssl, node->fd);
                SSL_set_connect_state(node->ssl);
            }
        }

        ssl_chk = SSL_do_handshake(node->ssl);
        if (ssl_chk == 1)
        {
            node->status = ABCDK_COMM_STATUS_STABLE;
        }
        else
        {
            ssl_err = SSL_get_error(node->ssl, ssl_chk);
            if (ssl_err == SSL_ERROR_WANT_WRITE)
            {
                chk = abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_OUTPUT, 0);
                if (chk != 0)
                    goto final_error;
            }
            else if (ssl_err == SSL_ERROR_WANT_READ)
            {
                chk = abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_INPUT, 0);
                if (chk != 0)
                    goto final_error;
            }
            else
                goto final_error;
        }
    }
#endif //HEADER_SSL_H

    /*修改保活参数，以防在远程断电的情况下本地无法检测到连接断开信号。*/
    if(node->status == ABCDK_COMM_STATUS_STABLE)
    {
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

    return;

final_error:

    /*修改超时，使用超时检测器关闭。*/
    abcdk_epollex_timeout(comm_ctx->epollex_ctx, node->fd, 1);
}

void abcdk_comm_cleanup()
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();

    abcdk_epollex_free(&comm_ctx->epollex_ctx);
}

int abcdk_comm_set_timeout(abcdk_comm_node *node,time_t timeout)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    int chk;

    assert(node != NULL);

    chk = abcdk_epollex_timeout(comm_ctx->epollex_ctx, node->fd, timeout);

    return chk;
}

int abcdk_comm_get_peername(abcdk_comm_node *node, abcdk_sockaddr_t *addr)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();

    assert(node != NULL && addr != NULL);

    *addr = node->remote;

    return 0;
}

void *abcdk_comm_set_userdata(abcdk_comm_node *node, void *opaque)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    void *old = NULL;

    assert(node != NULL);

    old = node->opaque;
    node->opaque = opaque;
    
    return old;
}

void *abcdk_comm_get_userdata(abcdk_comm_node *node)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    void *old = NULL;

    assert(node != NULL);

    old = node->opaque;
    
    return old;
}

ssize_t abcdk_comm_read(abcdk_comm_node *node, void *buf, size_t size)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    ssize_t rsize = 0,rsize_all = 0;
    int chk;

    assert(node != NULL && buf != NULL && size >0);

    /*仅消息循环线程拥有读权利。*/
    chk = abcdk_thread_leader_test(&node->input_user);
    assert(chk == 0);

    while (rsize_all < size)
    {
#ifdef HEADER_SSL_H
        if(node->ssl)
            rsize = SSL_read(node->ssl,ABCDK_PTR2PTR(void,buf,rsize_all),size-rsize_all);
        else 
#endif //HEADER_SSL_H
            rsize = recv(node->fd,ABCDK_PTR2PTR(void,buf,rsize_all),size-rsize_all,0);
        
        if(rsize <=0)
            break;
        
        rsize_all += rsize;
    }

    return rsize_all;
}

int abcdk_comm_read_watch(abcdk_comm_node *node, int done)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    int done_flag = 0;
    int chk;

    assert(node != NULL);

    if (done)
    {
        /*仅允许拥有读权利的线程释放读权利，其它线程只能注册读事件。*/
        chk = abcdk_thread_leader_quit(&node->input_user);
        if (chk == 0)
            done_flag = ABCDK_EPOLL_INPUT;
    }

    chk = abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_INPUT, done_flag);

    return chk;
}

ssize_t abcdk_comm_write(abcdk_comm_node *node, void *buf, size_t size)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    ssize_t wsize = 0,wsize_all = 0;

    assert(node != NULL && buf != NULL && size >0);

    while (wsize_all < size)
    {
#ifdef HEADER_SSL_H
        if(node->ssl)
            wsize = SSL_write(node->ssl,ABCDK_PTR2PTR(void,buf,wsize_all),size-wsize_all);
        else 
#endif //HEADER_SSL_H
            wsize = send(node->fd,ABCDK_PTR2PTR(void,buf,wsize_all),size-wsize_all,0);
        
        if(wsize <=0)
            break;
        
        wsize_all += wsize;
    }

    return wsize_all;
}

int abcdk_comm_write_watch(abcdk_comm_node *node)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    int chk;

    assert(node != NULL);

    chk = abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_OUTPUT, 0);

    return chk;
}

void _abcdk_tsl_event_cb(abcdk_comm_event_cb event_cb,abcdk_comm_node *node,uint32_t event)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    
    /*读权利绑定到线程。*/
    if(event == ABCDK_COMM_EVENT_INPUT)
        abcdk_thread_leader_vote(&node->input_user);

    /*重定向监听关闭事件ID。*/
    if ((event == ABCDK_COMM_EVENT_CLOSE) && (node->flag == ABCDK_COMM_FLAG_LISTEN))
        event = ABCDK_COMM_EVENT_LISTEN_CLOSE;

    /*重定向已连接事件ID。*/
    if ((event == ABCDK_COMM_EVENT_CONNECT) && (node->flag == ABCDK_COMM_FLAG_ACCPET))
        event = ABCDK_COMM_EVENT_ACCEPT;

    /*通知应用层处理事件。*/
    event_cb(node,event);
}

void abcdk_comm_loop(abcdk_comm_event_cb event_cb)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    abcdk_comm_node *node = NULL;
    abcdk_comm_node *node_sub = NULL;
    abcdk_epoll_event e = {0};
    int chk;
    
    assert(event_cb != NULL);

    while(1)
    {
        memset(&e,0,sizeof(abcdk_epoll_event));
        chk = abcdk_epollex_wait(comm_ctx->epollex_ctx, &e, 3000);
        if (chk < 0)
        {
            if(abcdk_atomic_load(&comm_ctx->loop_abort)==0)
                continue;
            else 
                break;
        }

        node = (abcdk_comm_node *)e.data.ptr;

        if(e.events & ABCDK_EPOLL_ERROR)
        {
            _abcdk_tsl_event_cb(event_cb,node,ABCDK_COMM_EVENT_CLOSE);

            /*释放引用，解除绑定，回收资源。*/
            abcdk_epollex_unref(comm_ctx->epollex_ctx,node->fd,e.events);
            abcdk_epollex_detach(comm_ctx->epollex_ctx,node->fd);
        }
        else
        {            
            if (e.events & ABCDK_EPOLL_INPUT)
            {
                if (node->flag == ABCDK_COMM_FLAG_LISTEN)
                {
                    /*接收新连接。*/
                    node_sub = _abcdk_comm_accept(node);

                    /*释放监听权利，并注册监听事件。*/
                    abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_INPUT, ABCDK_EPOLL_INPUT);

                    if (node_sub)
                    {
                        if (node_sub->status != ABCDK_COMM_STATUS_STABLE)
                        {
                            abcdk_comm_handshake(node_sub);
                            if (node_sub->status == ABCDK_COMM_STATUS_STABLE)
                                _abcdk_tsl_event_cb(event_cb,node_sub, ABCDK_COMM_EVENT_CONNECT);
                        }

                        /*释放读权利。*/
                        abcdk_epollex_mark(comm_ctx->epollex_ctx, node_sub->fd, 0, ABCDK_EPOLL_INPUT);
                    }
                    
                }
                else 
                {
                    if(node->status != ABCDK_COMM_STATUS_STABLE)
                    {
                        abcdk_comm_handshake(node);
                        if(node->status == ABCDK_COMM_STATUS_STABLE)
                            _abcdk_tsl_event_cb(event_cb,node,ABCDK_COMM_EVENT_CONNECT);

                        /*释放读权利。*/
                        abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, 0, ABCDK_EPOLL_INPUT);
                    }
                    else
                    {
                        _abcdk_tsl_event_cb(event_cb,node,ABCDK_COMM_EVENT_INPUT);
                        
                        /*数据的传输过程中，读权利的释放由应用层决定。*/
                    }
                }
            }
            
            if (e.events & ABCDK_EPOLL_OUTPUT)
            {
                if (node->status != ABCDK_COMM_STATUS_STABLE)
                {
                    abcdk_comm_handshake(node);
                    if (node->status == ABCDK_COMM_STATUS_STABLE)
                        _abcdk_tsl_event_cb(event_cb,node, ABCDK_COMM_EVENT_CONNECT);
                }
                else
                {
                    _abcdk_tsl_event_cb(event_cb,node, ABCDK_COMM_EVENT_OUTPUT);
                }
                
                /*释放写权利。*/
                abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, 0, ABCDK_EPOLL_OUTPUT);
            }

            /*释放引用计数。*/
            chk = abcdk_epollex_unref(comm_ctx->epollex_ctx,node->fd,e.events);
            assert(chk == 0);
        }
    }
}

void abcdk_comm_loop_abort()
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();

    abcdk_atomic_store(&comm_ctx->loop_abort,1);
}

int abcdk_comm_listen(abcdk_sockaddr_t *addr, SSL_CTX *ssl_ctx, void *opaque)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    abcdk_comm_node *node = NULL;
    epoll_data_t ep_data;
    int sock_flag = 1;
    int chk;

    assert(addr != NULL);

    node = _abcdk_comm_node_alloc();
    if (!node)
        return -1;

    node->flag = ABCDK_COMM_FLAG_LISTEN;
    node->status = ABCDK_COMM_STATUS_STABLE;
#ifdef HEADER_SSL_H
    node->ssl_ctx = ssl_ctx;
#endif //HEADER_SSL_H
    node->local = *addr;
    
    node->fd = abcdk_socket(addr->family, 0);
    if (node->fd < 0)
        goto final_error;

    /*端口复用，用于快速重启恢复。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(node->fd, SOL_SOCKET, SO_REUSEPORT, &sock_flag, 2);
    if (chk != 0)
        goto final_error;

    /*地址复用，用于快速重启恢复。*/
    sock_flag = 1;
    chk = abcdk_sockopt_option_int(node->fd, SOL_SOCKET, SO_REUSEADDR, &sock_flag, 2);
    if (chk != 0)
        goto final_error;

    if(addr->family == ABCDK_IPV6)
    {
        /*IPv6仅支持IPv6。*/
        sock_flag = 1;
        chk = abcdk_sockopt_option_int(node->fd, IPPROTO_IPV6, IPV6_V6ONLY, &sock_flag, 2);
        if (chk != 0)
            goto final_error;
    }

    chk = abcdk_bind(node->fd, &node->local);
    if (chk != 0) 
        goto final_error;

    chk = listen(node->fd, SOMAXCONN);
    if (chk != 0)
        goto final_error;

    ep_data.ptr = node;
    chk = abcdk_epollex_attach(comm_ctx->epollex_ctx,node->fd, &ep_data);
    if (chk != 0)
        goto final_error;
    
    /*关闭超时。*/
    abcdk_epollex_timeout(comm_ctx->epollex_ctx, node->fd, 0);
    abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_INPUT, 0);

    return 0;

final_error:

    _abcdk_comm_node_free(&node);

    return -1;
}

int abcdk_comm_connect(abcdk_sockaddr_t *addr, SSL_CTX *ssl_ctx, void *opaque)
{
    abcdk_comm_ctx *comm_ctx = _abcdk_comm_get_ctx();
    abcdk_comm_node *node = NULL;
    epoll_data_t ep_data;
    socklen_t addr_len;
    int sock_flag = 1;
    int chk;

    assert(addr != NULL);

    node = _abcdk_comm_node_alloc();
    if (!node)
        return -1;

    node->flag = ABCDK_COMM_FLAG_CLIENT;
    node->status = ABCDK_COMM_STATUS_SYNC;
    node->remote = *addr;
    node->opaque = opaque;
    
    node->fd = abcdk_socket(addr->family, 0);
    if (node->fd < 0)
        goto final_error;
#ifdef HEADER_SSL_H
    if(ssl_ctx)
    {
        node->ssl = abcdk_openssl_ssl_alloc(ssl_ctx);
        if(!node->ssl)
            goto final_error;
    }
#endif //HEADER_SSL_H

    chk = abcdk_fflag_add(node->fd,O_NONBLOCK);
    if(chk != 0 )
        goto final_error;

    addr_len = sizeof(abcdk_sockaddr_t);
    if(addr->family == AF_UNIX)
#ifdef SUN_LEN
        addr_len = SUN_LEN(&addr->addr_un);
#else 
        addr_len = offsetof(struct sockaddr_un,sun_path)+strlen(addr->addr_un.sun_path);
#endif

    chk = connect(node->fd, &addr->addr, addr_len);
    if(chk == 0)
        goto final;

    if (errno != EINPROGRESS && errno != EWOULDBLOCK && errno != EAGAIN)
        goto final_error;

final:

    ep_data.ptr = node;
    chk = abcdk_epollex_attach(comm_ctx->epollex_ctx, node->fd, &ep_data);
    if (chk != 0)
        goto final_error;

    abcdk_epollex_timeout(comm_ctx->epollex_ctx, node->fd, 30 * 1000);
    abcdk_epollex_mark(comm_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_OUTPUT, 0);

    return 0;

final_error:

    _abcdk_comm_node_free(&node);

    return -1;
}
