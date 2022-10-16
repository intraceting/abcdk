/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

typedef struct _abcdk_test_http
{
    int errcode;
    abcdk_tree_t *args;

    const char *listen;

    abcdk_comm_t *comm;
    abcdk_comm_node_t *listen_node;

} abcdk_test_http_t;

void _abcdk_test_http_accept_cb(abcdk_comm_node_t *node, int *result)
{
    *result = 0;
}

void _abcdk_test_http_event_cb(abcdk_comm_node_t *node, abcdk_http_request_t *req)
{
    size_t len = 0;
    const char *p, *val;
    for (int i = 0; i < 100; i++)
    {
        p = abcdk_http_request_env(req, i);
        if (!p)
            break;

        if (val = abcdk_http_match_env(p, "Content-Length"))
            len = strtol(val, NULL, 0);

        fprintf(stderr, "%s\n", p);
    }

#if 1

    if (len > 0)
    {
        abcdk_save("./test_http_upload.data", abcdk_http_request_body(req), len, 0);
    }

    abcdk_comm_message_t *file = abcdk_comm_message_mmap2("/tmp/aaa.txt", 0, 0);
    if (file)
    {
        abcdk_comm_message_t *msg = abcdk_comm_message_format(1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                                                            abcdk_http_status_desc(200), abcdk_comm_message_size(file));

        abcdk_http_send(node, msg);
        abcdk_http_send(node, file);
    }
    else
    {
        abcdk_comm_message_t *msg = abcdk_comm_message_format(1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                                                            abcdk_http_status_desc(404), 0);
        abcdk_http_send(node, msg);
    }
#else

    abcdk_comm_message_t *msg = abcdk_comm_message_format(1000, "RTSP/1.0 200 OK\r\nCSeq: 1\r\nPublic: OPTIONS, DESCRIBE, PLAY, PAUSE, SETUP, TEARDOWN, SET_PARAMETER, GET_PARAMETER\r\nDate:  Fri, Apr 10 2020 19:07:19 GMT\r\n\r\n");
    abcdk_http_send(node, msg);

#endif
}

void _abcdk_test_http_close_cb(abcdk_comm_node_t *node)
{
    char buf[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(node, NULL, buf);

    fprintf(stderr, "Disconnect: %s\n", buf);
}

#ifdef HEADER_SSL_H

int _abcdk_test_http_ssl_servername(SSL *ssl, int *ad, void *arg)
{
    return SSL_TLSEXT_ERR_OK;
}

int _abcdk_test_http_alpn_select_cb(SSL *ssl,
                                    const unsigned char **out,
                                    unsigned char *outlen,
                                    const unsigned char *in,
                                    unsigned int inlen,
                                    void *arg)
{
    for (int i = 0; i < inlen; i += in[i] + 1)
    {
        char buf[255] = {0};
        strncpy(buf,&in[i+1],(int)in[i]);
        fprintf(stderr, "SSL ALPN supported by client: %s\n", buf);
    }

    unsigned int srvlen;
   // unsigned char     srv[] = {"\x02h2\x08http/1.1\x08http/1.0\x08http/0.9"};
    //unsigned char srv[] = {"\x02h2"};
    unsigned char srv[] = {"\x08http/1.1"};

    srvlen = sizeof(srv) - 1;

    if (SSL_select_next_proto((unsigned char **)out, outlen, srv, srvlen, in, inlen) != OPENSSL_NPN_NEGOTIATED)
    {
        return SSL_TLSEXT_ERR_ALERT_FATAL;
    }

    fprintf(stderr, "SSL ALPN selected: %*s\n", (int)*outlen, *out);

    return SSL_TLSEXT_ERR_OK;
}

#endif

void _abcdk_test_http_work(abcdk_test_http_t *ctx)
{
    abcdk_sockaddr_t addr;
    ctx->listen = abcdk_option_get(ctx->args, "--listen", 0, "0.0.0.0:8080");

    ctx->comm = abcdk_comm_start(1, -1);

    ctx->listen_node = abcdk_http_alloc(ctx->comm, INT64_MAX, "/tmp/");
    abcdk_comm_set_userdata(ctx->listen_node, ctx);

    abcdk_sockaddr_from_string(&addr, ctx->listen, 1);

    SSL_CTX *server_ssl_ctx = NULL;

#ifdef HEADER_SSL_H

    const char *cafile = abcdk_option_get(ctx->args, "--ca-file", 0, NULL);
    const char *capath = abcdk_option_get(ctx->args, "--ca-path", 0, NULL);

    if (cafile || capath)
    {
        server_ssl_ctx = abcdk_openssl_ssl_ctx_alloc(1, cafile, capath, 0);

        abcdk_openssl_ssl_ctx_load_crt(server_ssl_ctx, abcdk_option_get(ctx->args, "--crt-file", 0, NULL),
                                       abcdk_option_get(ctx->args, "--key-file", 0, NULL),
                                       abcdk_option_get(ctx->args, "--key-pwd", 0, NULL));

        //  SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);

    //    SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER, NULL);
    //    SSL_CTX_set_tlsext_servername_callback(server_ssl_ctx,_abcdk_test_http_ssl_servername);
        SSL_CTX_set_alpn_select_cb(server_ssl_ctx, _abcdk_test_http_alpn_select_cb, NULL);

    //    SSL_CTX_set_options(server_ssl_ctx, SSL_OP_NO_TICKET);
    }

#endif

    abcdk_http_callback_t cb = {_abcdk_test_http_accept_cb, _abcdk_test_http_event_cb, _abcdk_test_http_close_cb};
    abcdk_http_listen(ctx->listen_node, server_ssl_ctx, &addr, &cb);

    while (getchar() != 'Q')
    {
        sleep(1);
    }

    abcdk_comm_unref(&ctx->listen_node);
    abcdk_comm_stop(&ctx->comm);
}

int abcdk_test_http(abcdk_tree_t *args)
{
    abcdk_test_http_t ctx = {0};

    ctx.args = args;

    _abcdk_test_http_work(&ctx);

    return ctx.errcode;
}