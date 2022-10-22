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

typedef struct _abcdk_test_h264
{
    int fd;
    abcdk_comm_queue_t *q;
}abcdk_test_h264_t;

void _abcdk_test_http_accept_cb(abcdk_comm_node_t *node, int *result)
{
    abcdk_test_h264_t *h = abcdk_heap_alloc(sizeof(abcdk_test_h264_t));

    h->fd = -1;
    h->q = abcdk_comm_queue_alloc();

    abcdk_comm_set_userdata(node,h);

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
        abcdk_save("./test_http_upload.data", abcdk_http_request_body(req,0), len, 0);
    }

    abcdk_object_t *file = abcdk_mmap2("/etc/issue", 0, 0, 0);
    if (file)
    {
        abcdk_http_send_format(node,1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\nContent-Type: %s; charset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                                                           abcdk_http_content_type_desc(".txt"),abcdk_http_status_desc(200), file->sizes[0]);

        abcdk_http_send_object(node, file);
    }
    else
    {
        abcdk_http_send_format(node,1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\ncharset=utf-8\r\nContent-Length: %lu\r\n\r\n",
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
                abcdk_save("./rtsp_ANNOUNCE.data", abcdk_http_request_body(req,0), len, 0);

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
        int len = abcdk_http_request_body_length(req);
        const void *p1 = abcdk_http_request_body(req,0);
        const void *p = abcdk_http_request_body(req,4);
        const void *p3 = abcdk_http_request_body(req,4+12);

        int c = abcdk_bloom_read_number(p1,4,8,8);

        abcdk_rtp_header_t t,t2={0};

        abcdk_rtp_header_deserialize(p,100,&t);

        char buf[100] = {0};
        abcdk_rtp_header_serialize(&t,buf,100);

        memcmp(buf,p,12);

        printf("c=%d,version=%d,padding=%d,extension=%d,csrc_len=%u,marker=%d,payload=%u,=seq_no=%u,timestamp=%u,ssrc=%u\n",
            c,
            t.version,t.padding,t.extension,
            t.csrc_len,t.marker,t.payload, t.seq_no,t.timestamp,t.ssrc);

        if(t.payload!=96)
            return;

        for(int i = 0;i<8;i++)
            printf("%d",abcdk_bloom_read_number(p3,1,i,1));
        printf("\n");
        
        abcdk_test_h264_t *h = (abcdk_test_h264_t*)abcdk_comm_get_userdata(node);

        if (h->fd <0)
            h->fd = abcdk_open("./test_rtsp_record.h264", 1, 0, 1);
     

        int chk = abcdk_rtp_h264_revert(p3,len-4-12,h->q);
        if(chk ==1)
        {
            while(1)
            {
                abcdk_comm_message_t*msg = abcdk_comm_queue_pop(h->q,1);
                if(!msg)
                    break;
                abcdk_write(h->fd ,"\0\0\0\1",4);
                abcdk_write(h->fd ,abcdk_comm_message_data(msg),abcdk_comm_message_offset(msg));

                abcdk_comm_message_unref(&msg);
            }
        }
    }
}

void _abcdk_test_http_close_cb(abcdk_comm_node_t *node)
{
    char buf[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(node, NULL, buf);

    abcdk_test_h264_t *h = (abcdk_test_h264_t*)abcdk_comm_get_userdata(node);
    if(h)
    {
        abcdk_closep(&h->fd);
        abcdk_comm_queue_free(&h->q);
        abcdk_heap_free(h);
    }

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

  //  abcdk_http_callback_t cb = {_abcdk_test_http_accept_cb, _abcdk_test_http_event_cb, _abcdk_test_http_close_cb};
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