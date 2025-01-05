/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

#if 0

typedef struct _abcdk_test_http
{
    int errcode;
    abcdk_option_t *args;

    const char *listen;

    abcdk_stcp_t *comm;
    abcdk_stcp_node_t *listen_node;

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

void _abcdk_test_http_msg_destroy_cb(void *msg)
{
    abcdk_receiver_t *msg_p = (abcdk_receiver_t *)msg;

    abcdk_receiver_unref(&msg_p);
}

void _abcdk_test_http_accept_cb(abcdk_stcp_node_t *node, int *result)
{
    abcdk_test_h264_t *h = abcdk_heap_alloc(sizeof(abcdk_test_h264_t));

    h->fd2 = h->fd = -1;
    h->q = abcdk_queue_alloc(_abcdk_test_http_msg_destroy_cb);
    h->q2 = abcdk_queue_alloc(_abcdk_test_http_msg_destroy_cb);

    abcdk_stcp_set_userdata(node, h);

    *result = 0;
}

void _abcdk_test_http_input_cb(abcdk_stcp_node_t *node, abcdk_http_receiver_t *req,int *next_proto)
{
    size_t len = 0;
    const char *p, *val;
    for (int i = 0; i < 100; i++)
    {
        p = abcdk_http_receiver_header(req, i);
        if (!p)
            break;

        if (val = abcdk_http_match_env(p, "Content-Length"))
            len = strtol(val, NULL, 0);

        fprintf(stderr, "%s\n", p);
    }

#if 1

    if (len > 0)
    {
        abcdk_save("./test_http_upload.data", abcdk_http_receiver_body(req, 0), len, 0);
    }

    abcdk_object_t *file = abcdk_mmap_filename("/etc/issue", 0, 0, 0);
    if (file)
    {
        abcdk_stcp_post_format(node, 1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\nContent-Type: %s; charset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                               abcdk_http_status_desc(200), abcdk_http_content_type_desc(".txt"),file->sizes[0]);

        abcdk_stcp_post(node, file);
    }
    else
    {
        abcdk_stcp_post_format(node, 1000, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\ncharset=utf-8\r\nContent-Length: %lu\r\n\r\n",
                               abcdk_http_status_desc(404), 0);
    }
#else

    abcdk_stcp_post_format(1000, "RTSP/1.0 200 OK\r\nCSeq: 1\r\nPublic: OPTIONS, DESCRIBE, PLAY, PAUSE, SETUP, TEARDOWN, SET_PARAMETER, GET_PARAMETER\r\nDate:  Fri, Apr 10 2020 19:07:19 GMT\r\n\r\n");

#endif
}

void _abcdk_test_rtsp_input_cb(abcdk_stcp_node_t *node, abcdk_http_receiver_t *req,int *next_proto)
{

    abcdk_test_h264_t *h = (abcdk_test_h264_t *)abcdk_stcp_get_userdata(node);

    // for (int i = 0; i < 100; i++)
    // {
    //     const char *p = abcdk_http_receiver_env(req, i);
    //     if (!p)
    //         break;

    //     fprintf(stderr, "%s\n", p);
    // }



    if (*next_proto == ABCDK_HTTP_RECEIVER_PROTO_RTSP)
    {
        const char *method_p = abcdk_http_receiver_header(req, 0);
        const char *cseq_p = abcdk_http_receiver_getenv(req, "cseq");
        const char *contlen_p = abcdk_http_receiver_getenv(req, "Content-Length");

        int cseq = strtol(cseq_p, NULL, 0);

        if (abcdk_strncmp(method_p, "OPTIONS", 7, 1) == 0)
        {
            abcdk_stcp_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_stcp_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_stcp_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
           // abcdk_stcp_post_format(node, 1000, "Public: OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY, PAUSE, GET_PARAMETER, SET_PARAMETER\r\n");
            abcdk_stcp_post_format(node, 1000, "Public: OPTIONS, DESCRIBE, TEARDOWN, PLAY, GET_PARAMETER, SET_PARAMETER\r\n");
            abcdk_stcp_post_format(node, 1000, "\r\n");
        }
        else if (abcdk_strncmp(method_p, "DESCRIBE", 8, 1) == 0)
        {
            abcdk_object_t *file = abcdk_mmap_filename("./rtsp_ANNOUNCE.data", 0, 0, 0);

            abcdk_stcp_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_stcp_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_stcp_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_stcp_post_format(node, 1000, "Content-Base: rtsp://192.168.1.188/h264/\r\n");
            abcdk_stcp_post_format(node, 1000, "Content-Type: application/sdp\r\n");
            abcdk_stcp_post_format(node, 1000, "Content-Length: %lu\r\n", file->sizes[0]);
            abcdk_stcp_post_format(node, 1000, "\r\n");
            abcdk_stcp_post(node, file);
        }
        else if (abcdk_strncmp(method_p, "ANNOUNCE", 8, 1) == 0)
        {
            int len = strtol(contlen_p, NULL, 0);
            if (len > 0)
                abcdk_save("./rtsp_ANNOUNCE.data", abcdk_http_receiver_body(req, 0), len, 0);

            abcdk_stcp_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_stcp_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_stcp_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_stcp_post_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_stcp_post_format(node, 1000, "Session: 123\r\n");
            abcdk_stcp_post_format(node, 1000, "\r\n");

            // printf("%s",abcdk_http_receiver_body(req,0));

            h->sdp = abcdk_rtsp_sdp_parse(abcdk_http_receiver_body(req, 0), len);
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
            abcdk_stcp_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_stcp_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_stcp_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_stcp_post_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_stcp_post_format(node, 1000, "Session: 123\r\n");
            abcdk_stcp_post_format(node, 1000, "Transport: RTP/AVP/TCP;unicast;interleaved=0-1;ssrc=00000000\r\n");
            abcdk_stcp_post_format(node, 1000, "x-Dynamic-Rate: 1\r\n");
            abcdk_stcp_post_format(node, 1000, "x-Transport-Options: late-tolerance=1.400000\r\n");
            abcdk_stcp_post_format(node, 1000, "\r\n");
        }
        else if (abcdk_strncmp(method_p, "RECORD", 5, 1) == 0)
        {
            abcdk_stcp_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            abcdk_stcp_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            abcdk_stcp_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            abcdk_stcp_post_format(node, 1000, "Server: test_rtsp\r\n");
            abcdk_stcp_post_format(node, 1000, "Session: 123\r\n");
            abcdk_stcp_post_format(node, 1000, "\r\n");

            *next_proto = ABCDK_HTTP_RECEIVER_PROTO_RTCP;
        }
        else if (abcdk_strncmp(method_p, "TEARDOWN", 8, 1) == 0)
        {
            // abcdk_stcp_post_format(node, 1000, "RTSP/1.0 %s\r\n", abcdk_http_status_desc(200));
            // abcdk_stcp_post_format(node, 1000, "CSeq: %d\r\n", cseq);
            // abcdk_stcp_post_format(node, 1000, "Date: Mon, Jul 21 2014 09:07:56 GMT\r\n");
            // abcdk_stcp_post_format(node, 1000, "Server: test_rtsp\r\n");
            // abcdk_stcp_post_format(node, 1000, "Session: 123\r\n");
            // abcdk_stcp_post_format(node, 1000, "\r\n");
        }
    }
    else
    {
        int len = abcdk_http_receiver_body_length(req);
        const void *p1 = abcdk_http_receiver_body(req, 0);
        const void *p = abcdk_http_receiver_body(req, 4);
        const void *p3 = abcdk_http_receiver_body(req, 4 + 12);

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
                    abcdk_receiver_t *msg = (abcdk_receiver_t *)abcdk_queue_pop(h->q, 1);
                    if (!msg)
                        break;
                    abcdk_write(h->fd, "\0\0\0\1", 4);
                    abcdk_write(h->fd, abcdk_receiver_data(msg), abcdk_receiver_offset(msg));

                    abcdk_receiver_unref(&msg);
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
                    abcdk_receiver_t *msg = (abcdk_receiver_t *)abcdk_queue_pop(h->q2, 1);
                    if (!msg)
                        break;

                    int hdr[7];
                    abcdk_aac_adts_header_t r;
                    r.syncword = 0xfff;
                    r.id = 0;
                    r.protection_absent = 1;
                    r.adts_buffer_fullness = 0x7ff;
                    r.aac_frame_length = abcdk_receiver_offset(msg) + 7;
                    r.channel_cfg = abcdk_aac_channels2config(atoi(h->a->encoder_param->pstrs[0]));
                    r.profile = 1;
                    r.sample_rate_index = abcdk_aac_sample_rates2index(h->a->clock_rate);

                    abcdk_aac_adts_header_serialize(&r, hdr, 7);
                    abcdk_write(h->fd2, hdr, 7);
                    abcdk_write(h->fd2, abcdk_receiver_data(msg), abcdk_receiver_offset(msg));

                    abcdk_receiver_unref(&msg);
                }
            }
        }
    }
}

void _abcdk_test_http_close_cb(abcdk_stcp_node_t *node)
{
    char buf[NAME_MAX] = {0};

    abcdk_stcp_get_sockaddr_str(node, NULL, buf);

    abcdk_test_h264_t *h = (abcdk_test_h264_t *)abcdk_stcp_get_userdata(node);
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

    ctx->comm = abcdk_stcp_start(-1,-1);

    ctx->listen_node = abcdk_http_alloc(ctx->comm,0,10000000000,"/tmp/");
    abcdk_stcp_set_userdata(ctx->listen_node, ctx);

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

      abcdk_http_callback_t cb = {NULL,_abcdk_test_http_accept_cb, _abcdk_test_http_input_cb,_abcdk_test_http_close_cb};
  //  abcdk_http_callback_t cb = {NULL,_abcdk_test_http_accept_cb, _abcdk_test_rtsp_input_cb, _abcdk_test_http_close_cb};
    abcdk_http_listen(ctx->listen_node, server_ssl_ctx, &addr, &cb);

    while (getchar() != 'Q')
    {
        sleep(1);
    }

    abcdk_stcp_unref(&ctx->listen_node);
    abcdk_stcp_stop(&ctx->comm);
}

int abcdk_test_http(abcdk_option_t *args)
{
    abcdk_test_http_t ctx = {0};

    ctx.args = args;

    _abcdk_test_http_work(&ctx);

    return ctx.errcode;
}

#elif 1

static void httpd_session_prepare_cb(void *opaque,abcdk_https_session_t **session,abcdk_https_session_t *listen)
{
    *session = abcdk_https_session_alloc((abcdk_https_t*)opaque);
}

static void httpd_session_accept_cb(void *opaque,abcdk_https_session_t *session,int *result)
{
    *result = 0;
}

static void httpd_session_ready_cb(void *opaque,abcdk_https_session_t *session)
{
  //  abcdk_https_session_set_timeout(session,10);
}

static void httpd_ssession_close_cb(void *opaque,abcdk_https_session_t *session)
{

}

static void httpd_request_cb(void *opaque, abcdk_https_stream_t *stream)
{
    // abcdk_trace_printf(LOG_INFO,"%s %s %s %s",
    // abcdk_https_request_header_get(stream,"Method"),
    // abcdk_https_request_header_get(stream,"Scheme"),
    // abcdk_https_request_header_get(stream,"Host"),
    // abcdk_https_request_header_get(stream,"Script"));

    // size_t len = 0;
    // const char* data = abcdk_https_request_body_get(stream,&len);
    // if(data)
    // {
    //     abcdk_trace_printf(LOG_INFO,"(%zd) %s",len,data);
    // }

    // int chk = abcdk_https_check_auth(stream);
    // if(chk != 0)
    //     return ;

    for(int i = 1;i<100;i++)
    {
        const char *p = abcdk_https_request_header_getline(stream,i);
        if(!p)
            break;

        fprintf(stderr,"{%s}\n",p);
    }

    char buf[200] = {0};
    memset(buf,'a',100);
    memset(buf+100,'b',100);

#if 1
    
    // abcdk_https_response_header(stream,200,100,
    //                 "Content-Length: %d\r\n"
    //                 "Content-Type: %s\r\n",
    //                 200,
    //                 "text/plain");


    abcdk_https_response_header_set(stream,"Content-Length","%d",200);
    abcdk_https_response_header_set(stream,"Content-Type","text/plain");
     abcdk_https_response_buffer(stream,buf,100);
     abcdk_https_response_buffer(stream,buf+100,100);
     abcdk_https_response(stream,NULL);
#elif 0
    //abcdk_https_response_buffer(stream,200,buf,200,"text/plain",NULL);

    int fd = abcdk_open("./aaaaa.mp4",0,0,0);
    if(fd>=0)
        abcdk_https_response_fd(stream,200,fd,"video/mp4",NULL);
    else 
        abcdk_https_response_nobody(stream,404,NULL,NULL);
    abcdk_closep(&fd);

#elif 0



#endif 
}

int abcdk_test_http(abcdk_option_t *args)
{
    abcdk_https_config_t cfg = {0};

    abcdk_logger_t *log_ctx = abcdk_logger_open2("/tmp/","test.http.log","test.http.%d.log",10,10,1,1);

    abcdk_trace_printf_set_callback(abcdk_logger_from_trace,log_ctx);

    abcdk_https_t *ctx = abcdk_https_create();

    cfg.opaque = ctx;
    cfg.pki_ca_file = abcdk_option_get(args,"--ca-file",0,NULL);
    cfg.pki_ca_path = abcdk_option_get(args,"--ca-path",0,NULL);
    // cfg.pki_cert_file = abcdk_option_get(args,"--cert-file",0,NULL);
    // cfg.pki_key_file = abcdk_option_get(args,"--key-file",0,NULL);
    cfg.req_max_size = abcdk_option_get_int(args,"--req-max-size",0,4*1024*1024);
    cfg.req_tmp_path = abcdk_option_get(args,"--req-tmp-path",0,NULL);
    cfg.enable_h2 = abcdk_option_get_int(args,"--enable-h2",0,0);
    cfg.name = abcdk_option_get(args,"--server-name",0,"test_http");
    cfg.realm = abcdk_option_get(args,"--server-realm",0,"abcdk");
    cfg.session_prepare_cb = httpd_session_prepare_cb;
    cfg.stream_request_cb = httpd_request_cb;
    cfg.session_ready_cb = httpd_session_ready_cb;
    cfg.auth_path = abcdk_option_get(args,"--auth-path",0,NULL);
    

    const char *listen = abcdk_option_get(args,"--listen",0,"ipv4://0.0.0.0:9999");

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_from_string(&addr,listen,0);


    abcdk_https_session_t *listen_ses_p = abcdk_https_session_alloc(ctx);

    abcdk_https_session_listen(listen_ses_p,&addr,&cfg);

    abcdk_https_session_unref(&listen_ses_p);

    abcdk_proc_wait_exit_signal(-1);

    abcdk_https_destroy(&ctx);

    abcdk_logger_close(&log_ctx);

    return 0;
}
#else 
int abcdk_test_http(abcdk_option_t *args)
{
    return 0;
}
#endif 
