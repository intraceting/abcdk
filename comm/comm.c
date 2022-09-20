/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "comm/comm.h"

/** 通讯环境。*/
typedef struct _abcdk_comm
{
    /** epollex 环境。*/
    abcdk_epollex_t *epollex;

    /** 工作线程。*/
    abcdk_thread_t *tids;

    /** 工人数量。*/
    int workers;

    /** 退出标志。0： 运行，!0：停止。*/
    volatile int exitflag;

} abcdk_comm_t;

/** 节点信息。*/
typedef struct _abcdk_comm_node
{
    /** 引用计数器。*/
    volatile int refcount;

    /** 
     * 通讯环境指针。
     * 
     * @warning 仅复制。
    */
    abcdk_comm_t *ctx;

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

    /** 事件回调函数指针。*/
    abcdk_comm_event_cb event_cb;

    /** 附加物。*/
    abcdk_object_t *append;

    /** 用户环境指针。*/
    abcdk_object_t *userdata;

} abcdk_comm_node_t;

void abcdk_comm_unref(abcdk_comm_node_t **node)
{
    abcdk_comm_node_t *node_p = NULL;

    if (!node || !*node)
        return;

    node_p = *node;
    *node = NULL;

    if (abcdk_atomic_fetch_and_add(&node_p->refcount, -1) != 1)
        return;

    assert(node_p->refcount == 0);

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_free(&node_p->ssl);
#endif //HEADER_SSL_H

    /*直接关闭，快速回收资源，不会处于time_wait状态。*/
    if (node_p->fd >= 0)
    {
        struct linger l = {1, 0};
        abcdk_socket_option_linger(node_p->fd, &l, 2);
    }

    abcdk_closep(&node_p->fd);
    abcdk_object_unref(&node_p->append);
    abcdk_object_unref(&node_p->userdata);
    abcdk_heap_free(node_p);
}

abcdk_comm_node_t *abcdk_comm_refer(abcdk_comm_node_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_comm_node_t *abcdk_comm_alloc(abcdk_comm_t *ctx)
{
    abcdk_comm_node_t *node = NULL;

    ABCDK_ASSERT(ctx != NULL,"通讯对象需要通讯环境才能被创建。");

    node = (abcdk_comm_node_t *)abcdk_heap_alloc(sizeof(abcdk_comm_node_t));
    if(!node)
        return NULL;

    node->refcount = 1;
    node->ctx = ctx;
    node->fd = -1;
    node->append = abcdk_object_alloc3(0,1);
    node->userdata = abcdk_object_alloc3(0,1);

    return node;
}

abcdk_object_t *abcdk_comm_append(abcdk_comm_node_t *node)
{
    assert(node != NULL);

    return abcdk_object_refer(node->append);
}

abcdk_object_t *abcdk_comm_userdata(abcdk_comm_node_t *node)
{
    assert(node != NULL);

    return abcdk_object_refer(node->userdata);
}


void *abcdk_comm_set_append(abcdk_comm_node_t *node,void *opaque)
{
    void *old = NULL;

    assert(node != NULL);

    old = node->append->pptrs[0];
    node->append->pptrs[0] = (uint8_t*)opaque;

    return old;
}

void *abcdk_comm_get_append(abcdk_comm_node_t *node)
{
    assert(node != NULL);

    return node->append->pptrs[0];
}

void *abcdk_comm_set_userdata(abcdk_comm_node_t *node,void *opaque)
{
    void *old = NULL;

    assert(node != NULL);

    old = node->userdata->pptrs[0];
    node->userdata->pptrs[0] = (uint8_t*)opaque;

    return old;
}

void *abcdk_comm_get_userdata(abcdk_comm_node_t *node)
{
    assert(node != NULL);

    return node->userdata->pptrs[0];
}

int abcdk_comm_set_timeout(abcdk_comm_node_t *node, time_t timeout)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    chk = abcdk_epollex_timeout(node->ctx->epollex, node->fd, timeout);

    return chk;
}

int abcdk_comm_get_sockaddr(abcdk_comm_node_t *node, abcdk_sockaddr_t *local,abcdk_sockaddr_t *remote)
{
    assert(node != NULL);

    if (local && node->local.family)
        abcdk_sockaddr_copy(&node->local,local);

    if (remote && node->remote.family)
        abcdk_sockaddr_copy(&node->remote,remote);

    return 0;
}

int abcdk_comm_get_sockaddr_str(abcdk_comm_node_t *node, char local[NAME_MAX],char remote[NAME_MAX])
{
    assert(node != NULL);

    if(local && node->local.family)
        abcdk_sockaddr_to_string(local,&node->local);

    if(remote && node->remote.family)
        abcdk_sockaddr_to_string(remote,&node->remote);

    return 0;
}

ssize_t abcdk_comm_recv(abcdk_comm_node_t *node, void *buf, size_t size)
{
    ssize_t rsize = 0,rsize_all = 0;
    int chk;

    assert(node != NULL && buf != NULL && size >0);

    /*仅消息循环线程拥有读权利。*/
    chk = abcdk_thread_leader_test(&node->input_user);
    ABCDK_ASSERT(chk == 0,"仅消息循环线程拥有读权利。");

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

int abcdk_comm_recv_watch(abcdk_comm_node_t *node)
{
    int done_flag = 0;
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    /*仅允许拥有读权利的线程释放读权利，其它线程只能注册读事件。*/
    chk = abcdk_thread_leader_quit(&node->input_user);
    if (chk == 0)
        done_flag = ABCDK_EPOLL_INPUT;

    chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_INPUT, done_flag);

    return chk;
}

ssize_t abcdk_comm_send(abcdk_comm_node_t *node, void *buf, size_t size)
{
    ssize_t wsize = 0,wsize_all = 0;

    assert(node != NULL && buf != NULL && size > 0);

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

ssize_t abcdk_comm_sendfile(abcdk_comm_node_t *node, int fd, off_t *offset, size_t count)
{
    ssize_t wsize = 0, wsize_all = 0;

    assert(node != NULL && fd >= 0 && count > 0);

#ifdef HEADER_SSL_H
    if (node->ssl)
    {
        ABCDK_ASSERT(0, "SSL不支持此接口。");
    }
    else
#endif // HEADER_SSL_H
    {
        wsize = sendfile(node->fd, fd, offset, count);
    }

    if (wsize > 0)
        wsize_all += wsize;

    return wsize_all;
}

int abcdk_comm_send_watch(abcdk_comm_node_t *node)
{
    int chk;

    assert(node != NULL);
    assert(node->ctx != NULL);

    chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_OUTPUT, 0);

    return chk;
}

void _abcdk_comm_cleanup_cb(epoll_data_t *data, void *opaque)
{
    abcdk_comm_t *ctx = (abcdk_comm_t *)opaque;
    abcdk_comm_node_t *node = NULL;

    node = (abcdk_comm_node_t *)data->ptr;
    abcdk_comm_unref(&node);
}

void _abcdk_comm_event_cb(abcdk_comm_node_t *node,uint32_t event, abcdk_comm_node_t *listen)
{
    /*读权利绑定到线程。*/
    if(event == ABCDK_COMM_EVENT_INPUT)
        abcdk_thread_leader_vote(&node->input_user);

    /*重定向监听关闭事件ID。*/
    if ((event == ABCDK_COMM_EVENT_CLOSE) && (node->flag == ABCDK_COMM_FLAG_LISTEN))
        event = ABCDK_COMM_EVENT_LISTEN_CLOSE;

    /*通知应用层处理事件。*/
    node->event_cb(node, event, listen);
}

abcdk_comm_node_t *_abcdk_comm_accept(abcdk_comm_node_t *listen)
{
    abcdk_comm_node_t *node_sub = NULL;
    epoll_data_t ep_data;
    int chk;

    node_sub = abcdk_comm_alloc(listen->ctx);
    if (!node_sub)
        return NULL;

    node_sub->flag = ABCDK_COMM_FLAG_ACCPET;
    node_sub->status = ABCDK_COMM_STATUS_SYNC;
    /*复制监听环境的回调函数指针。*/
    node_sub->event_cb = listen->event_cb;

    /*通知初始化。*/
    _abcdk_comm_event_cb(node_sub, ABCDK_COMM_EVENT_ACCEPT, listen);

#ifdef HEADER_SSL_H    
    if(listen->ssl_ctx)
    {
        node_sub->ssl = abcdk_openssl_ssl_alloc(listen->ssl_ctx);
        if(!node_sub->ssl)
            goto final_error;
    }
#endif //HEADER_SSL_H

    node_sub->fd = abcdk_accept(listen->fd, &node_sub->remote);
    if (node_sub->fd < 0)
        goto final_error;
    
    chk = abcdk_fflag_add(node_sub->fd,O_NONBLOCK);
    if(chk != 0 )
        goto final_error;

    ep_data.ptr = node_sub;
    chk = abcdk_epollex_attach(node_sub->ctx->epollex, node_sub->fd, &ep_data);
    if(chk != 0)
        goto final_error;

    abcdk_epollex_timeout(node_sub->ctx->epollex, node_sub->fd, 30*1000);

    return node_sub;

final_error:

    abcdk_comm_unref(&node_sub);
    
    return NULL;
}

void _abcdk_comm_handshake(abcdk_comm_node_t *node)
{
    socklen_t sock_len = 0;
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
            chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_OUTPUT, 0);
            if (chk != 0)
                goto final_error;
        }
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
                chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_OUTPUT, 0);
                if (chk != 0)
                    goto final_error;
            }
            else if (ssl_err == SSL_ERROR_WANT_READ)
            {
                chk = abcdk_epollex_mark(node->ctx->epollex, node->fd, ABCDK_EPOLL_INPUT, 0);
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
    abcdk_epollex_timeout(node->ctx->epollex, node->fd, 1);
}

void _abcdk_comm_perform(abcdk_comm_t *ctx,time_t timeout)
{
    abcdk_comm_node_t *node = NULL;
    abcdk_comm_node_t *node_sub = NULL;
    abcdk_epoll_event_t e = {0};
    int chk;

    memset(&e, 0, sizeof(abcdk_epoll_event_t));
    chk = abcdk_epollex_wait(ctx->epollex, &e, timeout);
    if (chk < 0)
        return;

    node = (abcdk_comm_node_t *)e.data.ptr;

    if (e.events & ABCDK_EPOLL_ERROR)
    {
        _abcdk_comm_event_cb(node, ABCDK_COMM_EVENT_CLOSE,NULL);

        /*释放引用。*/
        abcdk_epollex_unref(ctx->epollex, node->fd, e.events);
        /*解除绑定关系。*/
        abcdk_epollex_detach(ctx->epollex, node->fd);
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
                abcdk_epollex_mark(ctx->epollex, node->fd, ABCDK_EPOLL_INPUT, ABCDK_EPOLL_INPUT);

                if (node_sub)
                {
                    if (node_sub->status != ABCDK_COMM_STATUS_STABLE)
                    {
                        _abcdk_comm_handshake(node_sub);
                        if (node_sub->status == ABCDK_COMM_STATUS_STABLE)
                            _abcdk_comm_event_cb(node_sub, ABCDK_COMM_EVENT_CONNECT,NULL);
                    }

                    /*释放读权利。*/
                    abcdk_epollex_mark(ctx->epollex, node_sub->fd, 0, ABCDK_EPOLL_INPUT);
                }
            }
            else
            {
                if (node->status != ABCDK_COMM_STATUS_STABLE)
                {
                    _abcdk_comm_handshake(node);
                    if (node->status == ABCDK_COMM_STATUS_STABLE)
                        _abcdk_comm_event_cb(node, ABCDK_COMM_EVENT_CONNECT,NULL);

                    /*释放读权利。*/
                    abcdk_epollex_mark(ctx->epollex, node->fd, 0, ABCDK_EPOLL_INPUT);
                }
                else
                {
                    _abcdk_comm_event_cb(node, ABCDK_COMM_EVENT_INPUT,NULL);

                    /*在数据的传输过程中，读权利的释放由应用层决定，因此下面这句一定不要打开。*/
                    //abcdk_epollex_mark(ctx->epollex, node->fd, 0, ABCDK_EPOLL_INPUT);
                }
            }
        }

        if (e.events & ABCDK_EPOLL_OUTPUT)
        {
            if (node->status != ABCDK_COMM_STATUS_STABLE)
            {
                _abcdk_comm_handshake(node);
                if (node->status == ABCDK_COMM_STATUS_STABLE)
                    _abcdk_comm_event_cb(node, ABCDK_COMM_EVENT_CONNECT,NULL);
            }
            else
            {
                _abcdk_comm_event_cb(node, ABCDK_COMM_EVENT_OUTPUT,NULL);
            }

            /*无论连接状态如何，写权利必须内部释放，不能开放给应用层。*/
            abcdk_epollex_mark(ctx->epollex, node->fd, 0, ABCDK_EPOLL_OUTPUT);
        }

        /*释放引用计数。*/
        chk = abcdk_epollex_unref(ctx->epollex, node->fd, e.events);
        assert(chk == 0);
    }
}

void *_abcdk_comm_worker(void *args)
{
    abcdk_comm_t *ctx = (abcdk_comm_t *)args;

    /*每隔5秒检查一次，给退出检测留出时间。*/
    while (!abcdk_atomic_load(&ctx->exitflag))
        _abcdk_comm_perform(ctx, 3000);

    return NULL;
}

abcdk_comm_t * abcdk_comm_start(int workers)
{
    abcdk_comm_t *ctx = NULL;
    long nps = sysconf(_SC_NPROCESSORS_ONLN);
    cpu_set_t cpu_set;
    int chk;

    ctx = abcdk_heap_alloc(sizeof(abcdk_comm_t));
    if(!ctx)
        return NULL;

    ctx->epollex = abcdk_epollex_alloc(_abcdk_comm_cleanup_cb,ctx);
    ctx->workers = 0;
    ctx->exitflag = 0;

    /*如果未指定工作线程数，则使用CPU核心数。*/
    if (workers <= 0)
        workers = abcdk_align(nps/2,1);

    /*申请线程资源。*/
    ctx->tids = abcdk_heap_alloc(workers * sizeof(abcdk_thread_t));

    for (int i = 0; i < workers; i++)
    {
        ctx->tids[i].handle = 0;
        ctx->tids[i].routine = _abcdk_comm_worker;
        ctx->tids[i].opaque = ctx;
        chk = abcdk_thread_create(&ctx->tids[i], 1);
        if (chk != 0)
            goto final_error;

        /*线程启动，累加计数器。*/
        ctx->workers += 1;

        /*设置线程的CPU亲源性。*/
        CPU_ZERO(&cpu_set);
        CPU_SET((ctx->workers - i) % nps, &cpu_set);
        pthread_setaffinity_np(ctx->tids[i].handle, sizeof(cpu_set_t), &cpu_set);
    }

    return ctx;

final_error:
    
    abcdk_comm_stop(&ctx);

    return NULL;
}

void abcdk_comm_stop(abcdk_comm_t **ctx)
{
    abcdk_comm_t *ctx_p = NULL;

    if(!ctx || !*ctx)
        return;

    /*复制，清空。*/
    ctx_p = *ctx;
    *ctx = NULL;

    /*退出。*/
    abcdk_atomic_store(&ctx_p->exitflag, 1);

    /*回收线程资源。*/
    for (int i = 0; i < ctx_p->workers; i++)
        abcdk_thread_join(&ctx_p->tids[i]);

    abcdk_heap_free2((void **)&ctx_p->tids);
    abcdk_epollex_free(&ctx_p->epollex);
    abcdk_heap_free(ctx_p);
}

int abcdk_comm_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx,abcdk_sockaddr_t *addr, abcdk_comm_event_cb event_cb)
{
    abcdk_comm_node_t *node_p = NULL;
    epoll_data_t ep_data;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && addr != NULL && event_cb != NULL);

    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_comm_refer(node);

    node_p->flag = ABCDK_COMM_FLAG_LISTEN;
    node_p->status = ABCDK_COMM_STATUS_STABLE;
#ifdef HEADER_SSL_H
    node_p->ssl_ctx = ssl_ctx;
#endif //HEADER_SSL_H
    node_p->event_cb = event_cb;

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

    abcdk_comm_unref(&node_p);

    return -1;
}

int abcdk_comm_connect(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx,abcdk_sockaddr_t *addr, abcdk_comm_event_cb event_cb)
{
    abcdk_comm_node_t *node_p = NULL;
    epoll_data_t ep_data;
    socklen_t addr_len;
    int sock_flag = 1;
    int chk;

    assert(node != NULL && addr != NULL && event_cb != NULL);
    
    /*异步环境，首先得增加对象引用。*/
    node_p = abcdk_comm_refer(node);

    node_p->flag = ABCDK_COMM_FLAG_CLIENT;
    node_p->status = ABCDK_COMM_STATUS_SYNC;
    node_p->event_cb = event_cb;
    
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
#ifdef HEADER_SSL_H
    if(ssl_ctx)
    {
        node_p->ssl = abcdk_openssl_ssl_alloc(ssl_ctx);
        if(!node_p->ssl)
            goto final_error;
    }
#endif //HEADER_SSL_H

    chk = abcdk_fflag_add(node_p->fd,O_NONBLOCK);
    if(chk != 0 )
        goto final_error;


    chk = connect(node_p->fd, &node_p->remote.addr, addr_len);
    if(chk == 0)
        goto final;

    if (errno != EINPROGRESS && errno != EWOULDBLOCK && errno != EAGAIN)
        goto final_error;

final:

    /*节点加入epoll池中。在解除绑定关系前，节点不会被释放。*/
    ep_data.ptr = node_p;
    chk = abcdk_epollex_attach(node_p->ctx->epollex, node_p->fd, &ep_data);
    if (chk != 0)
        goto final_error;

    abcdk_epollex_timeout(node_p->ctx->epollex, node_p->fd, 30 * 1000);
    abcdk_epollex_mark(node_p->ctx->epollex, node_p->fd, ABCDK_EPOLL_OUTPUT, 0);

    return 0;

final_error:

    abcdk_comm_unref(&node_p);

    return -1;
}
