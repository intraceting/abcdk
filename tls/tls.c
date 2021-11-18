/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-tls/tls.h"

/**/
typedef struct _abcdk_tls_ctx
{
    abcdk_epollex_t *epollex_ctx;

} abcdk_tls_ctx;

/**/
typedef struct _abcdk_tls_node
{
    int flag;
#define ABCDK_TLS_FLAG_CLIENT   1
#define ABCDK_TLS_FLAG_LISTEN   2
#define ABCDK_TLS_FLAG_ACCPET   3

    int status;
#define ABCDK_TLS_STATUS_SYNC       1
#define ABCDK_TLS_STATUS_SSL_SYNC   2
#define ABCDK_TLS_STATUS_STABLE     3


    void *opaque;

    SSL_CTX *ssl_ctx;
    SSL *ssl;
    int fd;

    abcdk_sockaddr_t local;
    abcdk_sockaddr_t remote;

} abcdk_tls_node;

void _abcdk_tls_node_free(abcdk_tls_node **node)
{
    abcdk_tls_node *node_p = NULL;
    if (!node)
        return;

    node_p = *node;

    abcdk_openssl_ssl_free(&node_p->ssl);

    /*直接关闭，快速回收资源，不会处于time_wait状态。*/
    struct linger l = {1,0};
    abcdk_socket_option_linger(node_p->fd,&l,2);
    abcdk_closep(&node_p->fd);

    /*free and set NULL(0).*/
    abcdk_heap_free2((void **)node);
}

abcdk_tls_node *_abcdk_tls_node_alloc()
{
    return (abcdk_tls_node *)abcdk_heap_alloc(sizeof(abcdk_tls_node));
}

int _abcdk_tls_ctx_init(void *opaque)
{
    abcdk_tls_ctx *ctx = (abcdk_tls_ctx *)opaque;

    ctx->epollex_ctx = abcdk_epollex_alloc();
}

abcdk_tls_ctx *_abcdk_tls_get_ctx()
{
    static volatile int init = 0;
    static abcdk_tls_ctx ctx = {0};
    int chk;

    chk = abcdk_once(&init, _abcdk_tls_ctx_init, &ctx);
    assert(chk >= 0);

    return &ctx;
}

abcdk_tls_node *_abcdk_tls_accept(abcdk_tls_node *node)
{
    abcdk_tls_ctx *tls_ctx = _abcdk_tls_get_ctx();
    abcdk_tls_node *node_new = NULL;
    int sokc_flag = 1;
    int chk;

    node_new = _abcdk_tls_node_alloc();
    if (!node_new)
        return NULL;
    
    node_new->flag = ABCDK_TLS_FLAG_ACCPET;
    node_new->status = ABCDK_TLS_STATUS_STABLE;
    node_new->opaque = node->opaque;
    
    if(node->ssl_ctx)
    {
        node_new->ssl = abcdk_openssl_ssl_alloc(node->ssl_ctx);
        if(!node_new->ssl)
            goto final_error;

        node_new->status = ABCDK_TLS_STATUS_SSL_SYNC;
    }

    node_new->fd = abcdk_accept(node->fd, &node_new->remote);
    if (node_new->fd < 0)
        goto final_error;
    
    chk = abcdk_sockopt_option_int(node->fd, IPPROTO_TCP, TCP_NODELAY,&sokc_flag, 2);
    if(chk != 0)
        goto final_error;

    epoll_data_t data = {.ptr = node};
    chk = abcdk_epollex_attach(tls_ctx->epollex_ctx, node_new->fd, &data);
    if(chk != 0)
        goto final_error;

    chk = abcdk_epollex_timeout(tls_ctx->epollex_ctx, node_new->fd, 5*1000);
    if(chk != 0)
    {
        abcdk_epollex_detach(tls_ctx->epollex_ctx,node_new->fd);
        goto final_error;
    }

    return node_new;

final_error:

    _abcdk_tls_node_free(&node_new);
    
    return NULL;
}

void abcdk_tls_handshake(abcdk_tls_node *node)
{
    abcdk_tls_ctx *tls_ctx = _abcdk_tls_get_ctx();
    int sock_flag = 1;
    int ssl_chk;
    int ssl_err;
    int chk;

    if (node->status == ABCDK_TLS_STATUS_SYNC)
    {
        chk = abcdk_poll(node->fd, 0x02, 0);
        if (chk > 0)
        {
            node->status = ABCDK_TLS_STATUS_STABLE;
        }
        else
        {
            chk = abcdk_epollex_mark(tls_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_OUTPUT, 0);
            if (chk != 0)
                goto final_error;
        }
    }
    else if (node->status == ABCDK_TLS_STATUS_SSL_SYNC)
    {
        if (node->flag == ABCDK_TLS_FLAG_ACCPET)
        {
            if (SSL_get_fd(node->ssl) != node->fd)
            {
                SSL_set_fd(node->ssl, node->fd);
                SSL_set_accept_state(node->ssl);
            }
        }
        else if (node->flag == ABCDK_TLS_FLAG_CLIENT)
        {
            if (SSL_get_fd(node->ssl) != node->fd)
            {
                SSL_set_fd(node->ssl, node->fd);
                SSL_set_accept_state(node->ssl);
            }
        }

        ssl_chk = SSL_do_handshake(node->ssl);
        if (ssl_chk == 1)
        {
            node->status = ABCDK_TLS_STATUS_STABLE;
        }
        else
        {
            ssl_err = SSL_get_error(node->ssl, ssl_chk);
            if (ssl_err == SSL_ERROR_WANT_WRITE)
            {
                chk = abcdk_epollex_mark(tls_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_OUTPUT, 0);
                if (chk != 0)
                    goto final_error;
            }
            else if (ssl_err == SSL_ERROR_WANT_READ)
            {
                chk = abcdk_epollex_mark(tls_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_INPUT, 0);
                if (chk != 0)
                    goto final_error;
            }
            else
                goto final_error;
        }
    }

    /*修改保活参数，以防在远程断电的情况下，本地无法检测到连接断开信号。*/
    if(node->status == ABCDK_TLS_STATUS_STABLE)
    {
        /*开启keepalive属性*/
        sock_flag = 1;
        abcdk_sockopt_option_int(node->fd,SOL_SOCKET, SO_KEEPALIVE,&sock_flag,2);
        /*如该连接在60秒内没有任何数据往来，则进行探测。*/
        sock_flag = 60;
        abcdk_sockopt_option_int(node->fd,IPPROTO_TCP, TCP_KEEPIDLE,&sock_flag,2);
        /*探测时发包的时间间隔为5秒。*/
        sock_flag = 5;
        abcdk_sockopt_option_int(node->fd,IPPROTO_TCP, TCP_KEEPINTVL,&sock_flag,2);
        /*探测尝试的次数.如果第1次探测包就收到响应，则后2次的不再发。*/
        sock_flag = 3;
        abcdk_sockopt_option_int(node->fd,IPPROTO_TCP, TCP_KEEPCNT,&sock_flag,2);
    }

    return;

final_error:

    /*修改超时，使用超时检测器关闭。*/
    abcdk_epollex_timeout(tls_ctx->epollex_ctx, node->fd, 1);
}

void abcdk_tls_loop(void (*event_cb)(uint64_t tls, uint32_t event, void *opaque))
{
    abcdk_tls_ctx *tls_ctx = _abcdk_tls_get_ctx();
    abcdk_tls_node *node = NULL;
    abcdk_tls_node *node_new = NULL;
    abcdk_epoll_event e = {0};
    int chk;
    
    assert(event_cb != NULL);

    while(1)
    {
        memset(&e,0,sizeof(abcdk_epoll_event));
        int chk = abcdk_epollex_wait(tls_ctx->epollex_ctx, &e, -1);
        if (chk < 0)
            break;

        node = e.data.ptr;

        if(e.events & ABCDK_EPOLL_ERROR)
        {
            event_cb((uint64_t)node,ABCDK_TLS_EVENT_CLOSE,node->opaque);

            /*释放引用，解除绑定，回收资源。*/
            abcdk_epollex_unref(tls_ctx->epollex_ctx,node->fd,e.events);
            abcdk_epollex_detach(tls_ctx->epollex_ctx,node->fd);
            _abcdk_tls_node_free(&node);
        }
        else
        {            
            if (e.events & ABCDK_EPOLL_INPUT)
            {
                if (node->flag == ABCDK_TLS_FLAG_LISTEN)
                {
                    node_new = _abcdk_tls_accept(node);
                    if (node_new)
                    {
                        if (node_new->status != ABCDK_TLS_STATUS_STABLE)
                        {
                            abcdk_tls_handshake(node_new);
                            if (node_new->status == ABCDK_TLS_STATUS_STABLE)
                                event_cb((uint64_t)node_new, ABCDK_TLS_EVENT_CONNECT, node_new->opaque);
                        }
                    }

                    abcdk_epollex_mark(tls_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_INPUT, 0);
                }
                else 
                {
                    if(node->status != ABCDK_TLS_STATUS_STABLE)
                    {
                        abcdk_tls_handshake(node);
                        if(node->status == ABCDK_TLS_STATUS_STABLE)
                            event_cb((uint64_t)node,ABCDK_TLS_EVENT_CONNECT,node->opaque);
                    }
                    else
                    {
                        event_cb((uint64_t)node,ABCDK_TLS_EVENT_INPUT,node->opaque);
                    }
                }
            }
            else if (e.events & ABCDK_EPOLL_OUTPUT)
            {
                if (node->status != ABCDK_TLS_STATUS_STABLE)
                {
                    abcdk_tls_handshake(node);
                    if (node->status == ABCDK_TLS_STATUS_STABLE)
                        event_cb((uint64_t)node, ABCDK_TLS_EVENT_CONNECT, node->opaque);
                }
                else
                {
                    event_cb((uint64_t)node, ABCDK_TLS_EVENT_OUTPUT, node->opaque);
                }
            }
            
            chk = abcdk_epollex_mark(tls_ctx->epollex_ctx, node->fd, 0, e.events);
            chk = abcdk_epollex_unref(tls_ctx->epollex_ctx,node->fd,e.events);
            assert(chk == 0);
        }
    }
}

int abcdk_tls_listen(abcdk_sockaddr_t *addr, SSL_CTX *ssl_ctx, void *opaque)
{
    abcdk_tls_ctx *tls_ctx = _abcdk_tls_get_ctx();
    abcdk_tls_node *node = NULL;
    int sock_flag = 1;
    int chk;

    assert(addr != NULL);

    node = _abcdk_tls_node_alloc();
    if (!node)
        return -1;

    node->flag = ABCDK_TLS_FLAG_LISTEN;
    node->ssl_ctx = ssl_ctx;
    node->remote = *addr;
    node->status = ABCDK_TLS_STATUS_STABLE;

    node->fd = abcdk_socket(addr->family, 0);
    if (node->fd)
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

    chk = abcdk_bind(node->fd, &node->remote);
        if (chk != 0) goto final_error;

    chk = listen(node->fd, SOMAXCONN);
    if (chk != 0)
        goto final_error;

    epoll_data_t data = {.ptr = node};
    chk = abcdk_epollex_attach(tls_ctx->epollex_ctx,node->fd, &data);
    if (chk != 0)
        goto final_error;

    chk = abcdk_epollex_mark(tls_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_INPUT, 0);
    if (chk != 0)
    {
        abcdk_epollex_detach(tls_ctx->epollex_ctx,node->fd);
        goto final_error;
    }

final_error:

    _abcdk_tls_node_free(&node);

    return -1;
}