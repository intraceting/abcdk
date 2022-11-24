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
    int fd2;
    abcdk_queue_t *q;
    abcdk_queue_t *q2;

    abcdk_tree_t *sdp;
    abcdk_rtsp_sdp_media_base_t *v;
    abcdk_rtsp_sdp_media_base_t *a;

    int h264;
    int h265;

} abcdk_test_h264_t;

void _abcdk_test_http_msg_destroy_cb(const void *msg)
{
    abcdk_comm_message_t *msg_p = (abcdk_comm_message_t *)msg;

    abcdk_comm_message_unref(&msg_p);
}

void _abcdk_test_http_accept_cb(abcdk_comm_node_t *node, int *result)
{
    abcdk_test_h264_t *h = abcdk_heap_alloc(sizeof(abcdk_test_h264_t));

    h->fd2 = h->fd = -1;
    h->q = abcdk_queue_alloc(_abcdk_test_http_msg_destroy_cb);
    h->q2 = abcdk_queue_alloc(_abcdk_test_http_msg_destroy_cb);

    abcdk_comm_set_userdata(node, h);

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
        abcdk_save("./test_http_upload.data", abcdk_http_request_body(req, 0), len, 0);
    }

    abcdk_object_t *file = abcdk_mmap2("/etc/issue", 0, 0, 0);
    if (file)
    {
        abcdk_comm_post_format(node, 1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\nContent-Type: %s; charset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                               abcdk_http_content_type_desc(".txt"), abcdk_http_status_desc(200), file->sizes[0]);

        abcdk_comm_post(node, file);
    }
    else
    {
        abcdk_comm_post_format(node, 1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\ncharset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                               abcdk_http_status_desc(404), 0);
    }
#else

    abcdk_comm_post_format(1000, "RTSP/1.0 200 OK\r\nCSeq: 1\r\nPublic: OPTIONS, DESCRIBE, PLAY, PAUSE, SETUP, TEARDOWN, SET_PARAMETER, GET_PARAMETER\r\nDate:  Fri, Apr 10 2020 19:07:19 GMT\r\n\r\n");

#endif
}

void _abcdk_test_rtsp_event_cb(abcdk_comm_node_t *node, abcdk_http_request_t *req)
{

    abcdk_test_h264_t *h = (abcdk_test_h264_t *)abcdk_comm_get_userdata(node);

    for (int i = 0; i < 100; i++)
    {
        const char *p = abcdk_http_request_env(req, i);
        if (!p)
            break;

        fprintf(stderr, "%s\n", p);
    }

    const char *method_p = abcdk_http_request_env(req, 0);
    const char *cseq_p = abcdk_http_request_getenv(req, "cseq");
    const char *contlen_p = abcdk_http_request_getenv(req, "Content-Length");

    if (method_p)
    {

        int cseq = strtol(cseq_p, NULL, 0);

        if (abcdk_strncmp(method_p, "OPTIONS", 7, 1) == 0)
        {
            abcdk_comm_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_comm_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_comm_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_comm_post_format(node, 1000, "Public: OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY, PAUSE, GET_PARAMETER, SET_PARAMETER\r\n");
            abcdk_comm_post_format(node, 1000, "\r\n");
        }
        else if (abcdk_strncmp(method_p, "DESCRIBE", 8, 1) == 0)
        {
            abcdk_object_t *file = abcdk_mmap2("./rtsp_ANNOUNCE.data", 0, 0, 0);

            abcdk_comm_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_comm_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_comm_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_comm_post_format(node, 1000, "Content-Base: rtsp://192.168.1.188/h264/\r\n");
            abcdk_comm_post_format(node, 1000, "Content-Type: application/sdp\r\n");
            abcdk_comm_post_format(node, 1000, "Content-Length: %lu\r\n", file->sizes[0]);
            abcdk_comm_post_format(node, 1000, "\r\n");
            abcdk_comm_post(node, file);
        }
        else if (abcdk_strncmp(method_p, "ANNOUNCE", 8, 1) == 0)
        {
            int len = strtol(contlen_p, NULL, 0);
            if (len > 0)
                abcdk_save("./rtsp_ANNOUNCE.data", abcdk_http_request_body(req, 0), len, 0);

            abcdk_comm_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_comm_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_comm_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_comm_post_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_comm_post_format(node, 1000, "Session: 123\r\n");
            abcdk_comm_post_format(node, 1000, "\r\n");

            // printf("%s",abcdk_http_request_body(req,0));

            h->sdp = abcdk_rtsp_sdp_parse(abcdk_http_request_body(req, 0), len);
            abcdk_rtsp_sdp_dump(stderr, h->sdp);

            abcdk_rtsp_sdp_media_base_t *m = abcdk_rtsp_sdp_media_base_collect(h->sdp, 96);
            abcdk_rtsp_sdp_media_base_t *m1 = abcdk_rtsp_sdp_media_base_collect(h->sdp, 97);
            if (abcdk_strcmp(m->encoder->pstrs[0], "h264", 0) == 0)
                h->h264 = 1;
            if (abcdk_strcmp(m->encoder->pstrs[0], "h265", 0) == 0)
                h->h265 = 1;

            h->v = m;
            h->a = m1;
        }
        else if (abcdk_strncmp(method_p, "SETUP", 5, 1) == 0)
        {
            abcdk_comm_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_comm_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_comm_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_comm_post_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_comm_post_format(node, 1000, "Session: 123\r\n");
            abcdk_comm_post_format(node, 1000, "Transport: RTP/AVP/TCP;unicast;interleaved=0‐1;ssrc=00000000\r\n");
            abcdk_comm_post_format(node, 1000, "x‐Dynamic‐Rate: 1\r\n");
            abcdk_comm_post_format(node, 1000, "x‐Transport‐Options: late‐tolerance=1.400000\r\n");
            abcdk_comm_post_format(node, 1000, "\r\n");
        }
        else if (abcdk_strncmp(method_p, "RECORD", 5, 1) == 0)
        {
            abcdk_comm_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_comm_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_comm_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_comm_post_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_comm_post_format(node, 1000, "Session: 123\r\n");
            abcdk_comm_post_format(node, 1000, "\r\n");
        }
        else if (abcdk_strncmp(method_p, "TEARDOWN", 8, 1) == 0)
        {
            // abcdk_comm_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            // abcdk_comm_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            // abcdk_comm_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            // abcdk_comm_post_format(node, 1000, "Server: test_rtsp\r\n");
            // abcdk_comm_post_format(node, 1000, "Session: 123\r\n");
            // abcdk_comm_post_format(node, 1000, "\r\n");
        }
    }
    else
    {
        int len = abcdk_http_request_body_length(req);
        const void *p1 = abcdk_http_request_body(req, 0);
        const void *p = abcdk_http_request_body(req, 4);
        const void *p3 = abcdk_http_request_body(req, 4 + 12);

        abcdk_hexdump_option_t opt = {0};

        opt.flag = ABCDK_HEXDEMP_SHOW_ADDR | ABCDK_HEXDEMP_SHOW_CHAR;

        //   abcdk_hexdump(stderr,p1,4+12+16,0,&opt);

        int c = abcdk_bloom_read_number(p1, 4, 8, 8);

        abcdk_rtp_header_t t, t2 = {0};

        abcdk_rtp_header_deserialize(p, 100, &t);

        char buf[100] = {0};
        abcdk_rtp_header_serialize(&t, buf, 100);

        memcmp(buf, p, 12);

        printf("c=%d,version=%d,padding=%d,extension=%d,csrc_len=%u,marker=%d,payload=%u,=seq_no=%u,timestamp=%u,ssrc=%u\n",
            c,
            t.version,t.padding,t.extension,
            t.csrc_len,t.marker,t.payload, t.seq_no,t.timestamp,t.ssrc);

        if (t.payload == 96)
        {

            if (h->fd < 0)
            {
                h->fd = abcdk_open("./test_rtsp_record.h264", 1, 0, 1);

                if (h->h264)
                {
                    abcdk_write(h->fd, "\0\0\0\1", 4);
                    abcdk_write(h->fd, h->v->sprop_sps->pptrs[0], h->v->sprop_sps->sizes[0]);
                    abcdk_write(h->fd, "\0\0\0\1", 4);
                    abcdk_write(h->fd, h->v->sprop_pps->pptrs[0], h->v->sprop_pps->sizes[0]);
                }
                else if (h->h265)
                {
                    abcdk_write(h->fd, "\0\0\0\1", 4);
                    abcdk_write(h->fd, h->v->sprop_vps->pptrs[0], h->v->sprop_vps->sizes[0]);
                    abcdk_write(h->fd, "\0\0\0\1", 4);
                    abcdk_write(h->fd, h->v->sprop_sps->pptrs[0], h->v->sprop_sps->sizes[0]);
                    abcdk_write(h->fd, "\0\0\0\1", 4);
                    abcdk_write(h->fd, h->v->sprop_pps->pptrs[0], h->v->sprop_pps->sizes[0]);
                    if (h->v->sprop_sei)
                    {
                        abcdk_write(h->fd, "\0\0\0\1", 4);
                        abcdk_write(h->fd, h->v->sprop_sei->pptrs[0], h->v->sprop_sei->sizes[0]);
                    }
                }
            }

            int chk = -1;
            if (h->h264)
                chk = abcdk_rtp_h264_revert(p3, len - 4 - 12, h->q);
            else if (h->h265)
                chk = abcdk_rtp_hevc_revert(p3, len - 4 - 12, h->q);
            if (chk == 1)
            {
                while (1)
                {
                    abcdk_comm_message_t *msg = (abcdk_comm_message_t *)abcdk_queue_pop(h->q, 1);
                    if (!msg)
                        break;
                    abcdk_write(h->fd, "\0\0\0\1", 4);
                    abcdk_write(h->fd, abcdk_comm_message_data(msg), abcdk_comm_message_offset(msg));

                    abcdk_comm_message_unref(&msg);
                }
            }
        }
        else if (t.payload == 97)
        {
            if (h->fd2 < 0)
            {
                h->fd2 = abcdk_open("./test_rtsp_record.aac", 1, 0, 1);
            }
            int chk = -1;
            chk = abcdk_rtp_aac_revert(p3, len - 4 - 12, h->q2, 13, 3);
            if (chk == 1)
            {
                while (1)
                {
                    abcdk_comm_message_t *msg = (abcdk_comm_message_t *)abcdk_queue_pop(h->q2, 1);
                    if (!msg)
                        break;

                    int hdr[7];
                    abcdk_aac_adts_header_t r;
                    r.syncword = 0xfff;
                    r.id = 0;
                    r.protection_absent = 1;
                    r.adts_buffer_fullness = 0x7ff;
                    r.aac_frame_length = abcdk_comm_message_offset(msg) + 7;
                    r.channel_cfg = abcdk_aac_channels2config(atoi(h->a->encoder_param->pstrs[0]));
                    r.profile = 1;
                    r.sample_rate_index = abcdk_aac_sample_rates2index(h->a->clock_rate);

                    abcdk_aac_adts_header_serialize(&r, hdr, 7);
                    abcdk_write(h->fd2, hdr, 7);
                    abcdk_write(h->fd2, abcdk_comm_message_data(msg), abcdk_comm_message_offset(msg));

                    abcdk_comm_message_unref(&msg);
                }
            }
        }
    }
}

void _abcdk_test_http_fetch_cb(abcdk_comm_node_t *node)
{
    //    *delay = 1000;

    fprintf(stderr, "aaaa\n");
}

void _abcdk_test_http_close_cb(abcdk_comm_node_t *node)
{
    char buf[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(node, NULL, buf);

    abcdk_test_h264_t *h = (abcdk_test_h264_t *)abcdk_comm_get_userdata(node);
    if (h)
    {
        abcdk_closep(&h->fd);
        abcdk_closep(&h->fd2);
        abcdk_queue_free(&h->q);
        abcdk_queue_free(&h->q2);
        abcdk_tree_free(&h->sdp);
        abcdk_rtsp_sdp_media_base_free(&h->v);
        abcdk_rtsp_sdp_media_base_free(&h->a);
        abcdk_heap_free(h);
    }

    fprintf(stderr, "Disconnect: %s\n", buf);
}

void _abcdk_test_http_work(abcdk_test_http_t *ctx)
{
    abcdk_sockaddr_t addr;
    ctx->listen = abcdk_option_get(ctx->args, "--listen", 0, "0.0.0.0:8080");

    ctx->comm = abcdk_comm_start(-1);

    //ctx->listen_node = abcdk_http_alloc(ctx->comm, INT64_MAX, "/tmp/");
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

      //abcdk_http_callback_t cb = {_abcdk_test_http_accept_cb, _abcdk_test_http_event_cb,NULL,_abcdk_test_http_close_cb};
   // abcdk_http_callback_t cb = {_abcdk_test_http_accept_cb, _abcdk_test_rtsp_event_cb, _abcdk_test_http_fetch_cb, _abcdk_test_http_close_cb};
    //abcdk_http_listen(ctx->listen_node, server_ssl_ctx, &addr, &cb);

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