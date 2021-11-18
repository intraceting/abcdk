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
    /*
     * 0: client
     * 1: listen 
     * 2: accept 
    */
    int server;

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
        return NULL;

    node_p = *node;

    abcdk_openssl_ssl_free(&node_p->ssl);
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
    abcdk_tls_node *node_new = NULL;
    int sokc_flag = 1;

    node_new = _abcdk_tls_node_alloc();
    if (!node_new)
        return NULL;

    if(node->ssl_ctx)
    {
        node_new->ssl = abcdk_openssl_ssl_alloc(node->ssl_ctx);
        if(!node_new->ssl)
            goto final_error;
    }

    node_new->fd = abcdk_accept(node->fd, &node_new->remote);
    if (node_new->fd < 0)
        goto final_error;
    
    chk = abcdk_sockopt_option_int(c, IPPROTO_TCP, TCP_NODELAY,&sokc_flag, 2);
    if(chk != 0)
        goto final_error;

    node_new->server = 2;
    node_new->opaque = node->opaque;

    SSL_set_fd(node_new->ssl, node_new->fd);

    return node_new;

final_error:

    _abcdk_tls_node_free(&node_new);
    
    return NULL;
}

void abcdk_tls_loop(int (*event_cb)(uint64_t tls, uint32_t events, void *opaque))
{
    abcdk_tls_ctx *tls_ctx = NULL;
    abcdk_tls_node *node = NULL;
    abcdk_tls_node *node_new = NULL;
    abcdk_epoll_event e = {0};
    int chk;

    assert(event_cb != NULL);

    tls_ctx = _abcdk_tls_get_ctx();

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
            abcdk_epollex_detach(tls_ctx->epollex_ctx,node->fd）;
            _abcdk_tls_node_free(&node);
        }
        else if(1 == node->server)
        {
            while(1)
            {
                node_new = _abcdk_tls_node_alloc();
                if (!node_new)
                    break;

                node_new->fd = abcdk_accept(node->fd,&node_new->remote);
                if(node_new->fd<0)
                    break;

            }
        }
    }
}

int abcdk_tls_listen(abcdk_sockaddr_t *addr, SSL_CTX *ssl_ctx, void *opaque)
{
    abcdk_tls_ctx *tls_ctx = NULL;
    abcdk_tls_node *node = NULL;
    int sock_flag = 1;
    int chk;

    assert(addr != NULL);

    tls_ctx = _abcdk_tls_get_ctx();

    node = _abcdk_tls_node_alloc();
    if (!node)
        return -1;

    node->server = 1;
    node->ssl_ctx = ssl_ctx;
    node->remote = *addr;

    node->fd = abcdk_socket(addr->family, 0);
    if (node->fd)
        goto final_error;

    chk = abcdk_sockopt_option_int(node->fd, SOL_SOCKET, SO_REUSEPORT, &sock_flag, 2);
    if (chk != 0)
        goto final_error;

    chk = abcdk_sockopt_option_int(node->fd, SOL_SOCKET, SO_REUSEADDR, &sock_flag, 2);
    if (chk != 0)
        goto final_error;

    chk = abcdk_bind(node->fd, &node->remote);
        if (chk != 0) goto final_error;

    chk = listen(node->fd, SOMAXCONN);
    if (chk != 0)
        goto final_error;

    epoll_data_t data = {.ptr = node};
    chk = abcdk_epollex_attach(tls_ctx->epollex_ctx, &data);
    if (chk != 0)
        goto final_error;

    chk = abcdk_epollex_mark(tls_ctx->epollex_ctx, node->fd, ABCDK_EPOLL_INPUT, 0);
    if (chk != 0)
        goto final_error;

final_error:

    _abcdk_tls_node_free(&node);

    return -1;
}