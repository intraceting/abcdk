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

    abcdk_object_t *file = abcdk_mmap2("/etc/issue", 0, 0, 0);
    if (file)
    {
        abcdk_http_send_format(node,1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                                                            abcdk_http_status_desc(200), file->sizes[0]);

        abcdk_http_send_object(node, file);
    }
    else
    {
        abcdk_http_send_format(node,1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                                                            abcdk_http_status_desc(404), 0);
    }
#else

    abcdk_http_send_format(1000, "RTSP/1.0 200 OK\r\nCSeq: 1\r\nPublic: OPTIONS, DESCRIBE, PLAY, PAUSE, SETUP, TEARDOWN, SET_PARAMETER, GET_PARAMETER\r\nDate:  Fri, Apr 10 2020 19:07:19 GMT\r\n\r\n");

#endif
}

typedef struct _rtp_header
{
    int version;
    int padding;
    int extension;
    int csrc_len;
    int marker;
    int payload;
    int seq_no;
    int timestamp;
    int ssrc;
    int csrc;
    
} rtp_header_t;

uint64_t _readtonumber(const void *data, size_t size, off_t off, int bits)
{
    uint64_t num = 0;
    for (int i = 0; i < bits; i++)
        num = (num << 1) | abcdk_bloom_read((uint8_t *)data, size, off + i);

    return num;
}

void _abcdk_test_rtsp_event_cb(abcdk_comm_node_t *node, abcdk_http_request_t *req)
{
    
    for (int i = 0; i < 100; i++)
    {
        const char *p = abcdk_http_request_env(req, i);
        if (!p)
            break;

        fprintf(stderr, "%s\n", p);
    }


    const char *method_p = abcdk_http_request_env(req, 0);
    const char *cseq_p = abcdk_http_request_getenv(req,"cseq");
    const char *contlen_p = abcdk_http_request_getenv(req,"Content-Length");

    if (method_p)
    {

        int cseq = strtol(cseq_p, NULL, 0);

        if (abcdk_strncmp(method_p, "OPTIONS", 7, 1) == 0)
        {
            abcdk_http_send_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_http_send_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_http_send_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_http_send_format(node, 1000, "Public: OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY, PAUSE, GET_PARAMETER, SET_PARAMETER\r\n");
            abcdk_http_send_format(node, 1000, "\r\n");
        }
        else if (abcdk_strncmp(method_p, "DESCRIBE", 8, 1) == 0)
        {
            abcdk_object_t *file = abcdk_mmap2("./rtsp_ANNOUNCE.data", 0, 0, 0);

            abcdk_http_send_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_http_send_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_http_send_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_http_send_format(node, 1000, "Content-Base: rtsp://192.168.1.188/h264/\r\n");
            abcdk_http_send_format(node, 1000, "Content-Type: application/sdp\r\n");
            abcdk_http_send_format(node, 1000, "Content-Length: %lu\r\n", file->sizes[0]);
            abcdk_http_send_format(node, 1000, "\r\n");
            abcdk_http_send_object(node, file);
        }
        else if (abcdk_strncmp(method_p, "ANNOUNCE", 8, 1) == 0)
        {
            int len = strtol(contlen_p, NULL, 0);
            if (len > 0)
                abcdk_save("./rtsp_ANNOUNCE.data", abcdk_http_request_body(req), len, 0);

            abcdk_http_send_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_http_send_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_http_send_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_http_send_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_http_send_format(node, 1000, "Session: 123\r\n");
            abcdk_http_send_format(node, 1000, "\r\n");
        }
        else if (abcdk_strncmp(method_p, "SETUP", 5, 1) == 0)
        {
            abcdk_http_send_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_http_send_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_http_send_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_http_send_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_http_send_format(node, 1000, "Session: 123\r\n");
            abcdk_http_send_format(node, 1000, "Transport: RTP/AVP/TCP;unicast;interleaved=0‐1;ssrc=00000000\r\n");
            abcdk_http_send_format(node, 1000, "x‐Dynamic‐Rate: 1\r\n");
            abcdk_http_send_format(node, 1000, "x‐Transport‐Options: late‐tolerance=1.400000\r\n");
            abcdk_http_send_format(node, 1000, "\r\n");
        }
        else if (abcdk_strncmp(method_p, "RECORD", 5, 1) == 0)
        {
            abcdk_http_send_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_http_send_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_http_send_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_http_send_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_http_send_format(node, 1000, "Session: 123\r\n");
            abcdk_http_send_format(node, 1000, "\r\n");
        }
        else if(abcdk_strncmp(method_p, "TEARDOWN", 8, 1) == 0)
        {
            abcdk_http_send_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_http_send_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_http_send_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_http_send_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_http_send_format(node, 1000, "Session: 123\r\n");
            abcdk_http_send_format(node, 1000, "\r\n");
        }
    }
    else
    {
        const void *p = abcdk_http_request_body(req);

        rtp_header_t t;

        t.version = _readtonumber(p, 12, 0, 2);
        t.padding = _readtonumber(p, 12, 2, 1);
        t.extension = _readtonumber(p, 12, 3, 1);
        t.csrc_len = _readtonumber(p, 12, 4, 4);
        t.marker = _readtonumber(p, 12, 8, 1);
        t.payload = _readtonumber(p, 12, 9, 7);
        t.seq_no = _readtonumber(p, 12, 16, 16);
        t.timestamp = _readtonumber(p, 12, 32, 32);
        t.ssrc = _readtonumber(p, 12, 64, 32);

//        t.csrc = _readtonumber(p, 12, 97, 32);


        printf("version=%d,padding=%d,extension=%d,csrc_len=%u,marker=%d,payload=%u,=seq_no=%u,timestamp=%u,ssrc=%u\n",
            t.version,t.padding,t.extension,
            t.csrc_len,t.marker,t.payload, t.seq_no,t.timestamp,t.ssrc);
    }
}

void _abcdk_test_http_close_cb(abcdk_comm_node_t *node)
{
    char buf[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(node, NULL, buf);

    fprintf(stderr, "Disconnect: %s\n", buf);
}

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
        SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER, NULL);
    }

#endif

    //abcdk_http_callback_t cb = {_abcdk_test_http_accept_cb, _abcdk_test_http_event_cb, _abcdk_test_http_close_cb};
    abcdk_http_callback_t cb = {_abcdk_test_http_accept_cb, _abcdk_test_rtsp_event_cb, _abcdk_test_http_close_cb};
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