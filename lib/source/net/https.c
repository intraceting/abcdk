/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/net/https.h"

#ifdef HAVE_NGHTTP2
#include <nghttp2/nghttp2.h>
#endif // HAVE_NGHTTP2

#ifdef HAVE_LIBMAGIC
#include <magic.h>
#endif // HAVE_LIBMAGIC

/**简单的HTTP服务。*/
struct _abcdk_https
{
    /*IO对象*/
    abcdk_stcp_t *io_ctx;
    
}; // abcdk_https_t;

/**HTTP节点。*/
typedef struct _abcdk_https_node
{
    /*父级。*/
    abcdk_https_t *father;

    /*配置。*/
    abcdk_https_config_t cfg;

    /*标志。0 监听，1 服务端，2 客户端。*/
    int flag;

    /*协议。0：未选择 1：HTTP/1.1/1.0/0.9 2: HTTP/2 3：HTTP/3*/
    int protocol;

    /*安全方案*/
    int ssl_scheme;

    /*远程地址。*/
    char remote_addr[NAME_MAX];

    /*本机地址。*/
    char local_addr[NAME_MAX];
    
#ifdef NGHTTP2_H
    /*H2的回调和句柄。*/
    nghttp2_session_callbacks *h2_cbs;
    nghttp2_session *h2_handle;
#endif // NGHTTP2_H

    /*H2发送状态。*/
    int h2_send_busy;
    int64_t h2_send_want;
    int64_t h2_send_active;

    /*流容器。*/
    abcdk_map_t *stream_map;


    /*用户环境指针。*/
    void *userdata;

} abcdk_https_node_t;

/*流。*/
typedef struct _abcdk_https_stream_internal
{
    /*流ID。*/
    int id;

    /*协议。1：HTTP 4：TUNNEL*/
    int protocol;

    /*时间环境*/
    locale_t loc_ctx;

    /*IO节点。*/
    abcdk_stcp_node_t *io_node;

    /*上行数据。*/
    abcdk_receiver_t *updata;

    abcdk_object_t *method;
    abcdk_object_t *script;
    abcdk_object_t *version;
    abcdk_object_t *host;
    abcdk_object_t *scheme;

    /*H2的请求头部是否已经结束。*/
    int h2_req_hdr_end;

    /*H2输出流。*/
    abcdk_stream_t *h2_out;

    /*应答头。*/
    abcdk_option_t *rsp_hdr;

    /*H1应答头。*/
    abcdk_object_t *h1_rsp_hdrs;
    int h1_rsp_count;

    /*H2应答头。*/
#ifdef NGHTTP2_H
    nghttp2_nv h2_rsp_hdrs[100];
    int h2_rsp_count;
#endif //NGHTTP2_H

    /*应答头是否已发送。*/
    int rsp_hdr_sent;
    
    /*应答是否结束。*/
    int rsp_end;

    /*用户环境指针。*/
    void *userdata;

}abcdk_https_stream_internal_t;

static int64_t _abcdk_https_clock(uint8_t precision)
{
    return abcdk_time_clock2kind_with(CLOCK_MONOTONIC, precision);
}

static void _abcdk_https_stream_destructor_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_stcp_node_t *io_node_p;
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;

    io_node_p = (abcdk_stcp_node_t *)opaque;
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(io_node_p);
    stream_ctx_p = (abcdk_https_stream_internal_t *)obj->pptrs[ABCDK_MAP_VALUE];

    /*通知流执行析构。*/
    if(node_ctx_p->cfg.stream_destructor_cb)
        node_ctx_p->cfg.stream_destructor_cb(node_ctx_p->cfg.opaque,(abcdk_https_stream_t *)obj);

    if(stream_ctx_p->loc_ctx)
        freelocale(stream_ctx_p->loc_ctx);

    abcdk_receiver_unref(&stream_ctx_p->updata);
    abcdk_object_unref(&stream_ctx_p->method);
    abcdk_object_unref(&stream_ctx_p->script);
    abcdk_object_unref(&stream_ctx_p->version);
    abcdk_object_unref(&stream_ctx_p->host);
    abcdk_object_unref(&stream_ctx_p->scheme);
    abcdk_stream_destroy(&stream_ctx_p->h2_out);
    abcdk_stcp_unref(&stream_ctx_p->io_node);
    abcdk_object_unref(&stream_ctx_p->scheme);
    abcdk_object_unref(&stream_ctx_p->h1_rsp_hdrs);
    abcdk_option_free(&stream_ctx_p->rsp_hdr);
}

static void _abcdk_https_stream_construct_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_stcp_node_t *io_node_p;
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    
    io_node_p = (abcdk_stcp_node_t *)opaque;
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(io_node_p);
    stream_ctx_p = (abcdk_https_stream_internal_t *)obj->pptrs[ABCDK_MAP_VALUE];

    stream_ctx_p->id = *((int *)obj->pptrs[ABCDK_MAP_KEY]);
    stream_ctx_p->protocol = 1;
    stream_ctx_p->io_node = abcdk_stcp_refer(io_node_p);
    stream_ctx_p->h2_out = abcdk_stream_create();
    stream_ctx_p->loc_ctx = newlocale(LC_ALL_MASK,"en_US.UTF-8",NULL);

    /*通知流执行构造。*/
    if(node_ctx_p->cfg.stream_construct_cb)
        node_ctx_p->cfg.stream_construct_cb(node_ctx_p->cfg.opaque,(abcdk_https_stream_t *)obj);
}

static void _abcdk_https_stream_remove_cb(abcdk_object_t *stream, void *opaque)
{
    abcdk_stcp_node_t *io_node_p;
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;

    io_node_p = (abcdk_stcp_node_t *)opaque;
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(io_node_p);
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream->pptrs[ABCDK_MAP_VALUE];

    /*通知流已关闭。*/
    if(node_ctx_p->cfg.stream_close_cb)
        node_ctx_p->cfg.stream_close_cb(node_ctx_p->cfg.opaque,(abcdk_https_stream_t *)stream);
}

static void _abcdk_https_node_destroy_cb(void *userdata)
{
    abcdk_https_node_t *ctx;

    if (!userdata)
        return;

    ctx = (abcdk_https_node_t *)userdata;

    abcdk_map_destroy(&ctx->stream_map);

#ifdef NGHTTP2_H
    if (ctx->h2_handle)
        nghttp2_session_del(ctx->h2_handle);
    if (ctx->h2_cbs)
        nghttp2_session_callbacks_del(ctx->h2_cbs);
#endif // #NGHTTP2_H

}

static void _abcdk_https_log(abcdk_object_t *stream, uint32_t status)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    char new_tname[18] = {0}, old_tname[18] = {0};
    const char *user_agent_p,*refer_p;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    /*只记录HTTP日志。*/
    if (node_ctx_p->protocol != 1 && node_ctx_p->protocol != 2)
        return;

    if (status)
    {
        abcdk_trace_printf(LOG_INFO, "'%s'\n",abcdk_http_status_desc(status));
    }
    else
    {
        user_agent_p = abcdk_receiver_header_line_getenv(stream_ctx_p->updata,"User-Agent",':');
        refer_p = abcdk_receiver_header_line_getenv(stream_ctx_p->updata,"Referer",':');

        abcdk_trace_printf(LOG_INFO, "'%s' '%s' '%s' '%s' '%s'\n",
                                node_ctx_p->remote_addr,
                                stream_ctx_p->method->pstrs[0],
                                stream_ctx_p->script->pstrs[0],
                                (refer_p ? refer_p : "-"),
                                (user_agent_p ? user_agent_p : "-"));
    }

}


static void _abcdk_https_process(abcdk_object_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    const char *upgrade_val;
    int chk;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    _abcdk_https_log(stream,0);

    /*通知应用层数据到达。*/
    node_ctx_p->cfg.stream_request_cb(node_ctx_p->cfg.opaque,(abcdk_https_stream_t *)stream);

}

#ifdef NGHTTP2_H

static void _abcdk_https_process_2(abcdk_object_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (stream_ctx_p->protocol == 1)
    {
        ;
    }
    else if (stream_ctx_p->protocol == 4)
    {
        ;
    }

    _abcdk_https_process(stream);
}

static int _abcdk_https_h2_frame_recv_cb(nghttp2_session *session, const nghttp2_frame *frame, void *user_data)
{
    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;
    abcdk_https_node_t *node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    size_t remain = 0;
    int stream_id;

    stream_id = frame->hd.stream_id;

    /*过滤掉不需要的，但也不能返回失败。*/
    if(frame->hd.type != NGHTTP2_DATA && frame->hd.type != NGHTTP2_HEADERS)
        return 0;
    
    /*数据不完整，继续等待。*/
    if (!(frame->hd.flags & NGHTTP2_FLAG_END_STREAM))
        return 0;
    
    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, 0);
    if (!stream_p)
        return -1;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    if (stream_ctx_p->protocol == 1)
    {
        if (!stream_ctx_p->h2_req_hdr_end)
        {
            abcdk_receiver_append(stream_ctx_p->updata, "\r\n", 2, &remain);
            stream_ctx_p->h2_req_hdr_end = 1;
        }
    }
    else if (stream_ctx_p->protocol == 4)
    {

    }
    else
    {
        return -5;
    }

    _abcdk_https_process_2(stream_p);



    /*一定要回收。*/
    abcdk_receiver_unref(&stream_ctx_p->updata);

    return 0;
}

static int _abcdk_https_h2_data_chunk_recv_cb(nghttp2_session *session, uint8_t flags, int32_t stream_id, const uint8_t *data, size_t len, void *user_data)
{
    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;
    abcdk_https_node_t *node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    size_t remain = 0;
    int chk;

    /*更新流量控制窗口。*/
    nghttp2_session_consume(session, stream_id, len);

    /*如果没有设置自动更新窗口，需要手动提交 WINDOW_UPDATE 帧。*/
    nghttp2_submit_window_update(session, NGHTTP2_FLAG_NONE, stream_id, len);

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, 0);
    if (!stream_p)
        return -1;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    if (!stream_ctx_p->updata)
    {
        if(stream_ctx_p->protocol == 1)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_HTTP, node_ctx_p->cfg.req_max_size, node_ctx_p->cfg.req_tmp_path);
        else if(stream_ctx_p->protocol == 4)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM, 256*1024,NULL);
    }    

    if (!stream_ctx_p->updata)
        return -2;

    if (stream_ctx_p->protocol == 1)
    {
        if (!stream_ctx_p->h2_req_hdr_end)
        {
            abcdk_receiver_append(stream_ctx_p->updata, "\r\n", 2, &remain);
            stream_ctx_p->h2_req_hdr_end = 1;
        }

        chk = abcdk_receiver_append(stream_ctx_p->updata, data, len, &remain);
        if (chk < 0)
            return -3;
        else if (chk == 0) /*数据包不完整，继续接收。*/
            return 0;

    }
    else if (stream_ctx_p->protocol == 4)
    {
        chk = abcdk_receiver_append(stream_ctx_p->updata, data, len, &remain);
        if (chk < 0)
            return -3;
        else if (chk == 0) /*数据包不完整，继续接收。*/
            return 0;
    }
    else
    {
        return -5;
    }

    return 0;
}

static int _abcdk_https_h2_header_cb(nghttp2_session *session, const nghttp2_frame *frame,
                                            const uint8_t *name, size_t namelen,
                                            const uint8_t *value, size_t valuelen,
                                            uint8_t flags, void *user_data)
{
    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;
    abcdk_https_node_t *node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    int stream_id;
    size_t remain = 0;

    stream_id = frame->hd.stream_id;

    /*过滤掉不需要的，但也不能返回失败。*/
    if(frame->hd.type != NGHTTP2_HEADERS)
        return 0;

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, 0);
    if (!stream_p)
        return -1;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    if (!stream_ctx_p->updata)
    {
        if(stream_ctx_p->protocol == 1)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_HTTP, node_ctx_p->cfg.req_max_size, node_ctx_p->cfg.req_tmp_path);
        else if(stream_ctx_p->protocol == 4)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM, 256*1024,NULL);
    }    

    if (!stream_ctx_p->updata)
        return -2;
  
    if (abcdk_strcmp(":method", name, 0) == 0)
    {
        stream_ctx_p->method = abcdk_object_copyfrom(value, valuelen);
        return 0;
    }
    else if (abcdk_strcmp(":scheme", name, 0) == 0)
    {
        stream_ctx_p->scheme = abcdk_object_copyfrom(value, valuelen);
        return 0;
    }
    else if (abcdk_strcmp(":authority", name, 0) == 0)
    {
        stream_ctx_p->host = abcdk_object_copyfrom(value, valuelen);
        return 0;
    }
    else if (abcdk_strcmp(":path", name, 0) == 0)
    {
        stream_ctx_p->script = abcdk_object_copyfrom(value, valuelen);
        return 0;
    }
    else
    {
      //  abcdk_trace_printf(LOG_INFO, "%s: {%s}{%d}{%d}\r\n", name, value,(frame->hd.flags),(flags));

        abcdk_receiver_append(stream_ctx_p->updata, name, namelen, &remain);
        abcdk_receiver_append(stream_ctx_p->updata, ": ", 2, &remain);
        abcdk_receiver_append(stream_ctx_p->updata, value, valuelen, &remain);
        abcdk_receiver_append(stream_ctx_p->updata, "\r\n", 2, &remain);
        if (stream_ctx_p->h2_req_hdr_end = (flags & NGHTTP2_FLAG_END_HEADERS))
            abcdk_receiver_append(stream_ctx_p->updata, "\r\n", 2, &remain);
    }

    return 0;
}

static int _abcdk_https_h2_begin_headers_cb(nghttp2_session *session, const nghttp2_frame *frame, void *user_data)
{
    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;
    abcdk_https_node_t *node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    int stream_id;

    stream_id = frame->hd.stream_id;

    /*过滤掉不需要的，但也不能返回失败。*/
    if (frame->hd.type != NGHTTP2_HEADERS || frame->headers.cat != NGHTTP2_HCAT_REQUEST)
        return 0;

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, sizeof(abcdk_https_stream_internal_t));
    if (!stream_p)
        return -1;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    /*删除过时的。*/
    abcdk_object_unref(&stream_ctx_p->method);
    abcdk_object_unref(&stream_ctx_p->script);
    abcdk_object_unref(&stream_ctx_p->version);
    abcdk_object_unref(&stream_ctx_p->host);
    abcdk_object_unref(&stream_ctx_p->scheme);
    stream_ctx_p->rsp_hdr_sent = 0;
    stream_ctx_p->rsp_end = 0;
    stream_ctx_p->h2_rsp_count = 0;

    return 0;
}

static int _abcdk_https_h2_stream_close_cb(nghttp2_session *session, int32_t stream_id, uint32_t error_code, void *user_data)
{
    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;
    abcdk_https_node_t *node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    /*移除这个节点。*/
    abcdk_map_remove2(node_ctx_p->stream_map, &stream_id);

    return 0;
}

static ssize_t _abcdk_https_h2_send_cb(nghttp2_session *session, const uint8_t *data, size_t length, int flags, void *user_data)
{
    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;
    abcdk_https_node_t *node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);
    int chk;

    /*记录活动时间。*/
    node_ctx_p->h2_send_active = _abcdk_https_clock(0);

    /*等待线路空闲时再发送。*/
    if(node_ctx_p->h2_send_busy)
    {
        /*不忙了。*/
        node_ctx_p->h2_send_busy = 0;
        return NGHTTP2_ERR_WOULDBLOCK; 
    }

    /*假设线路即将忙碌。*/
    node_ctx_p->h2_send_busy = 1; 
    chk = abcdk_stcp_post_buffer(node, data, length);
    if (chk != 0)
        return NGHTTP2_ERR_EOF;

    return length;
}

static ssize_t _abcdk_https_h2_response_read_cb(nghttp2_session *session, int32_t stream_id, uint8_t *buf, size_t length,
                                                uint32_t *data_flags, nghttp2_data_source *source, void *user_data)
{
    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;
    abcdk_https_node_t *node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    ssize_t rlen;

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, 0);
    if (!stream_p)
        return -1;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    
    for (int i = 0; i < 2; i++)
    {
        rlen = abcdk_stream_read(stream_ctx_p->h2_out, buf, length);
        if (rlen > 0)
            return rlen;

        /*如果应用层已经应答结束后，标记已结束。*/
        if (stream_ctx_p->rsp_end)
        {
            *data_flags |= NGHTTP2_DATA_FLAG_EOF;
            return 0;
        }

        /*通知流空闲。*/
        if (node_ctx_p->cfg.stream_output_cb)
            node_ctx_p->cfg.stream_output_cb(node_ctx_p->cfg.opaque, (abcdk_https_stream_t *)stream_p);
    }

    /*如果缓存区没有待发送数据，并且应用层也没有数据准备好，才会走到这里。*/
    return NGHTTP2_ERR_DEFERRED;
}

#endif // NGHTTP2_H

static void _abcdk_https_process_1(abcdk_object_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    const char *line_p;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (stream_ctx_p->protocol == 1)
    {
        line_p = abcdk_receiver_header_line(stream_ctx_p->updata, 0);
        if (!line_p)
            goto ERR;

        /*解析请求行。*/
        abcdk_http_parse_request_header0(line_p, &stream_ctx_p->method, &stream_ctx_p->script, &stream_ctx_p->version);

        line_p = abcdk_receiver_header_line_getenv(stream_ctx_p->updata,"Host",':');
        if (!line_p)
            goto ERR;

        stream_ctx_p->host = abcdk_object_copyfrom(line_p,strlen(line_p));

        if (node_ctx_p->ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI)
            stream_ctx_p->scheme = abcdk_object_copyfrom("https", 5);
        else
            stream_ctx_p->scheme = abcdk_object_copyfrom("http", 4);
    }
    else if (stream_ctx_p->protocol == 4)
    {
        ;
    }

    _abcdk_https_process(stream);

    return;

ERR:

    abcdk_stcp_set_timeout(stream_ctx_p->io_node,-1);

}

static void _abcdk_https_request_1(abcdk_stcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_object_t *stream_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    int stream_id = 0;
    const char *upgrade_val;
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, sizeof(abcdk_https_stream_internal_t));
    if (!stream_p)
        goto ERR;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    if (!stream_ctx_p->updata)
    {
        if (stream_ctx_p->protocol == 1)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_HTTP, node_ctx_p->cfg.req_max_size, node_ctx_p->cfg.req_tmp_path);
        else if (stream_ctx_p->protocol == 4)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM, 256*1024, NULL);
        else
            goto ERR;
    }

    if (!stream_ctx_p->updata)
        goto ERR;

    chk = abcdk_receiver_append(stream_ctx_p->updata, data, size, remain);
    if (chk < 0)
    {
        *remain = 0;
        goto ERR;
    }
    else if (chk == 0) /*数据包不完整，继续接收。*/
        return;

    /*删除过时的。*/
    abcdk_object_unref(&stream_ctx_p->method);
    abcdk_object_unref(&stream_ctx_p->script);
    abcdk_object_unref(&stream_ctx_p->version);
    abcdk_object_unref(&stream_ctx_p->host);
    abcdk_object_unref(&stream_ctx_p->scheme);
    abcdk_object_unref(&stream_ctx_p->h1_rsp_hdrs);
    abcdk_option_free(&stream_ctx_p->rsp_hdr);
    stream_ctx_p->rsp_hdr_sent = 0;
    stream_ctx_p->rsp_end = 0;
    stream_ctx_p->h1_rsp_count = 0;

    /**/
    _abcdk_https_process_1(stream_p);
    
    /*一定要回收。*/
    abcdk_receiver_unref(&stream_ctx_p->updata);

    /*No Error.*/
    return;

ERR:

    abcdk_stcp_set_timeout(node, -1);
}

static void _abcdk_https_request_2(abcdk_stcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_https_node_t *node_ctx_p;
    ssize_t chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;
    
#ifdef NGHTTP2_H

    chk = nghttp2_session_mem_recv(node_ctx_p->h2_handle, data, size);
    if (chk < 0)
        goto ERR;
    else if (chk < size)
        *remain = size - chk;

#endif // NGHTTP2_H

    /*No Error.*/
    return;

ERR:

    abcdk_stcp_set_timeout(node, -1);
}

static void _abcdk_https_output_1(abcdk_stcp_node_t *node)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_object_t *stream_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    int stream_id = 0;
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, 0);
    if (!stream_p)
        return;

    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    /*通知流空闲。*/
    if(node_ctx_p->cfg.stream_output_cb)
        node_ctx_p->cfg.stream_output_cb(node_ctx_p->cfg.opaque,(abcdk_https_stream_t *)stream_p);
}


static void _abcdk_https_output_2(abcdk_stcp_node_t *node)
{
    abcdk_https_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

#ifdef NGHTTP2_H

    /*记录活动时间。*/
    node_ctx_p->h2_send_want = _abcdk_https_clock(0);

    /*如果线路长时间不活动，表示没数据要发送。*/
    if(node_ctx_p->h2_send_active != 0 && node_ctx_p->h2_send_want - node_ctx_p->h2_send_active > 5)
        return;

    /*控制流量。*/
    if (nghttp2_session_want_write(node_ctx_p->h2_handle))
    {
        /*把缓存数据串行化，并通过回调发送出去。*/
        chk = nghttp2_session_send(node_ctx_p->h2_handle);
        if (chk < 0)
        {
            abcdk_stcp_set_timeout(node, -1);
            return;
        }
    }

    /*继续监听，尽可能快的发送数据到客户端。*/
    abcdk_stcp_send_watch(node);

#endif //NGHTTP2_H
}

static void _abcdk_https_prepare_cb(abcdk_stcp_node_t **node, abcdk_stcp_node_t *listen)
{
    abcdk_stcp_node_t *node_p;
    abcdk_https_node_t *listen_ctx_p, *node_ctx_p;
    int chk;

    listen_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(listen);

    listen_ctx_p->cfg.session_prepare_cb(listen_ctx_p->cfg.opaque,(abcdk_https_session_t **)&node_p,(abcdk_https_session_t *)listen);
    if (!node_p)
        return;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->cfg = listen_ctx_p->cfg;
    node_ctx_p->flag = 1;
    node_ctx_p->protocol = 0;
    node_ctx_p->ssl_scheme = listen_ctx_p->cfg.ssl_scheme;

    /*准备完毕，返回。*/
    *node = node_p;
}

static void _abcdk_https_event_accept(abcdk_stcp_node_t *node, int *result)
{
    abcdk_https_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);


    /*默认：允许。*/
    *result = 0;

    if(node_ctx_p->cfg.session_accept_cb)
        node_ctx_p->cfg.session_accept_cb(node_ctx_p->cfg.opaque, (abcdk_https_session_t*)node, result);
    
    if(*result != 0)
        abcdk_trace_printf(LOG_WARNING, "禁止客户端(%s)连接到本机(%s)。", node_ctx_p->remote_addr, node_ctx_p->local_addr);
}

static void _abcdk_https_event_connect(abcdk_stcp_node_t *node)
{
    abcdk_https_node_t *node_ctx_p;
    char proto[256] = {0};
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    if (node_ctx_p->ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI)
    {
        abcdk_stcp_ssl_get_alpn_selected(node, proto);
        if (node_ctx_p->protocol == 0)
        {
            if (abcdk_strncmp("http", proto, 4, 0) == 0)
                node_ctx_p->protocol = 1;
            else if (abcdk_strcmp("h2", proto, 0) == 0)
                node_ctx_p->protocol = 2;
        }
    }

    abcdk_trace_printf(LOG_WARNING, "本机(%s)与远端(%s)的连接已建立。",node_ctx_p->local_addr,node_ctx_p->remote_addr);
    
    /*设置超时。*/
    abcdk_stcp_set_timeout(node, 180);

    /*如果未选择协议，则使用默认协议。*/
    if(node_ctx_p->protocol == 0)
        node_ctx_p->protocol = 1;

    if (node_ctx_p->protocol == 2)
    {
#ifdef NGHTTP2_H
        nghttp2_session_callbacks_new(&node_ctx_p->h2_cbs);
        nghttp2_session_callbacks_set_on_frame_recv_callback(node_ctx_p->h2_cbs, _abcdk_https_h2_frame_recv_cb);
        nghttp2_session_callbacks_set_on_data_chunk_recv_callback(node_ctx_p->h2_cbs, _abcdk_https_h2_data_chunk_recv_cb);
        nghttp2_session_callbacks_set_on_header_callback(node_ctx_p->h2_cbs, _abcdk_https_h2_header_cb);
        nghttp2_session_callbacks_set_on_begin_headers_callback(node_ctx_p->h2_cbs, _abcdk_https_h2_begin_headers_cb);
        nghttp2_session_callbacks_set_on_stream_close_callback(node_ctx_p->h2_cbs, _abcdk_https_h2_stream_close_cb);
        nghttp2_session_callbacks_set_send_callback(node_ctx_p->h2_cbs, _abcdk_https_h2_send_cb);

        nghttp2_session_server_new(&node_ctx_p->h2_handle, node_ctx_p->h2_cbs, node);

        nghttp2_settings_entry iv[] = {
            {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS, 100},
            {NGHTTP2_SETTINGS_MAX_HEADER_LIST_SIZE, 100},
            {NGHTTP2_SETTINGS_MAX_FRAME_SIZE, 65535},
            {NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE, 16 * 1024 * 1024}};

        /*必须要设置。*/
        nghttp2_submit_settings(node_ctx_p->h2_handle, NGHTTP2_FLAG_NONE, iv, 4);
#endif // NGHTTP2_H
    }

    if(node_ctx_p->cfg.session_ready_cb)
        node_ctx_p->cfg.session_ready_cb(node_ctx_p->cfg.opaque, (abcdk_https_session_t*)node);

    /*已连接到远端，注册读写事件。*/
    abcdk_stcp_recv_watch(node);
    abcdk_stcp_send_watch(node);
}

static void _abcdk_https_event_output(abcdk_stcp_node_t *node)
{
    abcdk_https_node_t *node_ctx_p;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    if(node_ctx_p->protocol == 1)
    {
        _abcdk_https_output_1(node);
    }
    else if( node_ctx_p->protocol == 2)
    {
        _abcdk_https_output_2(node);
    }
}

static void _abcdk_https_event_close(abcdk_stcp_node_t *node)
{
    abcdk_https_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    if (node_ctx_p->flag == 0)
    {
        abcdk_trace_printf(LOG_INFO, "监听关闭，忽略。");
        return;
    }

    abcdk_trace_printf(LOG_INFO, "本机(%s)与远端(%s)的连接已断开。",node_ctx_p->local_addr,node_ctx_p->remote_addr);

    /*一定要在这里释放，否则在单路复用时，由于多次引用的原因会使当前链路得不到释放。*/
    abcdk_map_destroy(&node_ctx_p->stream_map);

    if(node_ctx_p->cfg.session_close_cb)
        node_ctx_p->cfg.session_close_cb(node_ctx_p->cfg.opaque,(abcdk_https_session_t*)node);
}

static void _abcdk_https_event_cb(abcdk_stcp_node_t *node, uint32_t event, int *result)
{
    abcdk_https_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    if(!node_ctx_p->remote_addr[0])
        abcdk_stcp_get_sockaddr_str(node,NULL,node_ctx_p->remote_addr);
    if(!node_ctx_p->local_addr[0])
        abcdk_stcp_get_sockaddr_str(node,node_ctx_p->local_addr,NULL);


    if (event == ABCDK_STCP_EVENT_ACCEPT)
    {
        _abcdk_https_event_accept(node,result);
    }
    else if (event == ABCDK_STCP_EVENT_CONNECT)
    {
        _abcdk_https_event_connect(node);
    }
    else if (event == ABCDK_STCP_EVENT_INPUT)
    {
    }
    else if (event == ABCDK_STCP_EVENT_OUTPUT)
    {
        _abcdk_https_event_output(node);
    }
    else if (event == ABCDK_STCP_EVENT_CLOSE || event == ABCDK_STCP_EVENT_INTERRUPT)
    {
        _abcdk_https_event_close(node);
    }
}


static void _abcdk_https_input_cb(abcdk_stcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_https_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    if (node_ctx_p->protocol == 1)
    {
        _abcdk_https_request_1(node, data, size, remain);
    }
    else if (node_ctx_p->protocol == 2)
    {
        _abcdk_https_request_2(node, data, size, remain);
    }
}

void abcdk_https_session_unref(abcdk_https_session_t **session)
{
    abcdk_stcp_unref((abcdk_stcp_node_t**)session);
}

abcdk_https_session_t *abcdk_https_session_refer(abcdk_https_session_t *src)
{
    return (abcdk_https_session_t*)abcdk_stcp_refer((abcdk_stcp_node_t*)src);
}

abcdk_https_session_t *abcdk_https_session_alloc(abcdk_https_t *ctx)
{
    abcdk_stcp_node_t *node_p;
    abcdk_https_node_t *node_ctx_p;
    abcdk_object_t *stream_p;
    abcdk_https_stream_internal_t *stream_ctx_p;

    assert(ctx != NULL);

    node_p = abcdk_stcp_alloc(ctx->io_ctx, sizeof(abcdk_https_node_t), _abcdk_https_node_destroy_cb);
    if (!node_p)
        return NULL;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->father = ctx;
    node_ctx_p->protocol = 0;
    node_ctx_p->stream_map = abcdk_map_create(1);
    if(!node_ctx_p->stream_map)
        goto ERR;

    node_ctx_p->stream_map->construct_cb = _abcdk_https_stream_construct_cb;
    node_ctx_p->stream_map->destructor_cb = _abcdk_https_stream_destructor_cb;
    node_ctx_p->stream_map->remove_cb = _abcdk_https_stream_remove_cb;
    node_ctx_p->stream_map->opaque = node_p;

    return (abcdk_https_session_t*)node_p;

ERR:

    abcdk_https_session_unref((abcdk_https_session_t**)&node_p);

    return NULL;
}

void *abcdk_https_session_get_userdata(abcdk_https_session_t *session)
{
    abcdk_stcp_node_t *node_p;
    abcdk_https_node_t *node_ctx_p;

    assert(session != NULL);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node_p);

    return node_ctx_p->userdata;
}

void *abcdk_https_session_set_userdata(abcdk_https_session_t *session,void *userdata)
{
    abcdk_stcp_node_t *node_p;
    abcdk_https_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node_p);

    old_userdata = node_ctx_p->userdata;
    node_ctx_p->userdata = userdata;
    
    return old_userdata;
}

const char *abcdk_https_session_get_address(abcdk_https_session_t *session, int remote)
{
    abcdk_stcp_node_t *node_p;
    abcdk_https_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL);

    node_p = (abcdk_stcp_node_t *)session;
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node_p);

    if (remote)
        return node_ctx_p->remote_addr;
    else
        return "";
}

void abcdk_https_session_set_timeout(abcdk_https_session_t *session,time_t timeout)
{
    abcdk_stcp_node_t *node_p;
    abcdk_https_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL && timeout >= 1);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node_p);

    abcdk_stcp_set_timeout(node_p,timeout);
}


int abcdk_https_session_listen(abcdk_https_session_t *session,abcdk_sockaddr_t *addr,abcdk_https_config_t *cfg)
{
    abcdk_stcp_node_t *node_p;
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_stcp_config_t asio_cfg = {0};
    int chk;

    assert(session != NULL && addr != NULL && cfg != NULL);
    assert(cfg->ssl_scheme == ABCDK_STCP_SSL_SCHEME_RAW || cfg->ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI);
    assert(cfg->session_prepare_cb != NULL && cfg->stream_request_cb != NULL);
    assert(cfg->req_max_size > 1024);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->cfg = *cfg;
    node_ctx_p->flag = 0;
    node_ctx_p->protocol = 0;

    asio_cfg.ssl_scheme = cfg->ssl_scheme;
    asio_cfg.pki_ca_file = cfg->pki_ca_file;
    asio_cfg.pki_ca_path = cfg->pki_ca_path;
    asio_cfg.pki_chk_crl = cfg->pki_chk_crl;
    asio_cfg.pki_use_cert = cfg->pki_use_cert;
    asio_cfg.pki_use_key = cfg->pki_use_key;

    if (cfg->ssl_scheme == ABCDK_STCP_SSL_SCHEME_PKI)
    {
        /*Default HTTP/1.1*/
        asio_cfg.pki_next_proto = "\x08http/1.1";

#ifdef NGHTTP2_H
        if (cfg->enable_h2)
        {
            asio_cfg.pki_next_proto = "\x02h2\x08http/1.1";
            asio_cfg.pki_cipher_list = "AES128-GCM-SHA256:CHACHA20-POLY1305-SHA256";
        }
#endif // NGHTTP2_H

    }

    asio_cfg.prepare_cb = _abcdk_https_prepare_cb;
    asio_cfg.event_cb = _abcdk_https_event_cb;
    asio_cfg.input_cb = _abcdk_https_input_cb;

    asio_cfg.bind_addr = *addr;

    chk = abcdk_stcp_listen(node_p,&asio_cfg);
    if(chk == 0)
        return 0;

    return -1;
}

void abcdk_https_destroy(abcdk_https_t **ctx)
{
    abcdk_https_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_stcp_stop(ctx_p->io_ctx);
    abcdk_stcp_destroy(&ctx_p->io_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_https_t *abcdk_https_create()
{
    abcdk_https_t *ctx;
    int chk;

    ctx = (abcdk_https_t *)abcdk_heap_alloc(sizeof(abcdk_https_t));
    if (!ctx)
        return NULL;

    ctx->io_ctx = abcdk_stcp_create(sysconf(_SC_NPROCESSORS_ONLN));
    if (!ctx->io_ctx)
        goto ERR;

    return ctx;
ERR:

    abcdk_https_destroy(&ctx);

    return NULL;
}

/** 释放流。*/
void abcdk_https_unref(abcdk_https_stream_t **stream)
{
    abcdk_object_t *stream_p;

    if(!stream ||!*stream)
        return;

    stream_p = (abcdk_object_t *)*stream;
    *stream = NULL;

    abcdk_object_unref(&stream_p);
}

/** 引用流。*/
abcdk_https_stream_t *abcdk_https_refer(abcdk_https_stream_t *src)
{
    abcdk_object_t *stream_p;

    assert(src != NULL);

    stream_p = (abcdk_object_t *)src;

    return (abcdk_https_stream_t *)abcdk_object_refer(stream_p);
}

abcdk_https_session_t *abcdk_https_get_session(abcdk_https_stream_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    return (abcdk_https_session_t*)stream_ctx_p->io_node;
}

void *abcdk_https_get_userdata(abcdk_https_stream_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    return stream_ctx_p->userdata;
}

void *abcdk_https_set_userdata(abcdk_https_stream_t *stream,void *userdata)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    void *old_userdata;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    old_userdata = stream_ctx_p->userdata;
    stream_ctx_p->userdata = userdata;
    
    return old_userdata;

}


const char *abcdk_https_request_header_get(abcdk_https_stream_t *stream, const char *key)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    /*HTTP协议有头部数据。*/
    if (stream_ctx_p->protocol != 1)
        goto ERR;

    if (abcdk_strcmp("method", key, 0) == 0)
        return stream_ctx_p->method->pstrs[0];
    else if (abcdk_strcmp("script", key, 0) == 0)
        return stream_ctx_p->script->pstrs[0];
    else if (abcdk_strcmp("scheme", key, 0) == 0)
        return stream_ctx_p->scheme->pstrs[0];
    else if (abcdk_strcmp("host", key, 0) == 0)
        return stream_ctx_p->host->pstrs[0];
    else
        return abcdk_receiver_header_line_getenv(stream_ctx_p->updata, key, ':');

ERR:

    return NULL;
}

const char* abcdk_https_request_header_getline(abcdk_https_stream_t *stream,int line)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;

    assert(stream != NULL && line >= 1 && line < 100);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    /*HTTP协议有头部数据。*/
    if (stream_ctx_p->protocol != 1)
        goto ERR;
    
    return abcdk_receiver_header_line(stream_ctx_p->updata, line);
ERR:

    return NULL;
}

const char *abcdk_https_request_body_get(abcdk_https_stream_t *stream, size_t *len)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    const void *body_p;
    size_t body_l;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    body_l = abcdk_receiver_body_length(stream_ctx_p->updata);
    if (body_l <= 0)
        return NULL;

    if (len)
        *len = body_l;

    body_p = abcdk_receiver_body(stream_ctx_p->updata, 0);

    return body_p;
}

static int _abcdk_https_rsp_hdr_dump_cb(const char *key, const char *value, void *opaque)
{
    abcdk_https_stream_internal_t *stream_ctx_p = (abcdk_https_stream_internal_t *)opaque;
    abcdk_https_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    /*状态码不在这里处理。*/
    if (abcdk_strcmp("Status", key, 0) == 0)
        return 0;

    if (node_ctx_p->protocol == 1)
    {
        chk = snprintf(stream_ctx_p->h1_rsp_hdrs->pstrs[0] + stream_ctx_p->h1_rsp_count, stream_ctx_p->h1_rsp_hdrs->sizes[0] - stream_ctx_p->h1_rsp_count, "%s: %s\r\n", key, value);
        if (chk <= 0)
            return -1;

        stream_ctx_p->h1_rsp_count += chk;
    }
    else if (node_ctx_p->protocol == 2)
    {
#ifdef NGHTTP2_H
        stream_ctx_p->h2_rsp_hdrs[stream_ctx_p->h2_rsp_count].name = (uint8_t*)key;
        stream_ctx_p->h2_rsp_hdrs[stream_ctx_p->h2_rsp_count].namelen = strlen(key);
        stream_ctx_p->h2_rsp_hdrs[stream_ctx_p->h2_rsp_count].value = (uint8_t*)value;
        stream_ctx_p->h2_rsp_hdrs[stream_ctx_p->h2_rsp_count].valuelen = strlen(value);

        stream_ctx_p->h2_rsp_count +=1;
#endif //NGHTTP2_H
    }
    else
    {
        return -1;
    }

    return 0;
}

static int _abcdk_https_response_header_h1(abcdk_https_stream_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    abcdk_option_iterator_t it = {0};
    uint32_t status;
    int chk;

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (node_ctx_p->protocol != 1)
        return -1;

    abcdk_object_unref(&stream_ctx_p->h1_rsp_hdrs);
    stream_ctx_p->h1_rsp_hdrs = abcdk_object_alloc2(256*1024);
    if(!stream_ctx_p->h1_rsp_hdrs)
        return -2;

    status = abcdk_option_get_int(stream_ctx_p->rsp_hdr,"Status",0,0);
    if(status == 0)
        return -3;
    
    /*构造状态行。*/
    chk = snprintf(stream_ctx_p->h1_rsp_hdrs->pstrs[0] + stream_ctx_p->h1_rsp_count, stream_ctx_p->h1_rsp_hdrs->sizes[0] - stream_ctx_p->h1_rsp_count, "HTTP/1.1 %s\r\n", abcdk_http_status_desc(status));
    if (chk <= 0)
        return -4;

    stream_ctx_p->h1_rsp_count += chk;

    /*遍历其它行。*/
    it.opaque = stream_ctx_p;
    it.dump_cb = _abcdk_https_rsp_hdr_dump_cb;
    abcdk_option_scan(stream_ctx_p->rsp_hdr,&it);

    /*构造结束行。*/
    chk = snprintf(stream_ctx_p->h1_rsp_hdrs->pstrs[0] + stream_ctx_p->h1_rsp_count, stream_ctx_p->h1_rsp_hdrs->sizes[0] - stream_ctx_p->h1_rsp_count, "\r\n");
    if (chk <= 0)
        return -4;

    stream_ctx_p->h1_rsp_count += chk;

    /*标记头部长度。*/
    stream_ctx_p->h1_rsp_hdrs->sizes[0] = stream_ctx_p->h1_rsp_count;
    chk = abcdk_stcp_post(stream_ctx_p->io_node, stream_ctx_p->h1_rsp_hdrs,1);
    if (chk != 0)
        return -2;

    /*发送成功就托管了。*/
    stream_ctx_p->h1_rsp_hdrs = NULL;

    return 0;
}

static int _abcdk_https_response_header_h2(abcdk_https_stream_t *stream)
{
#ifdef NGHTTP2_H
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    abcdk_option_iterator_t it = {0};
    nghttp2_data_provider data_prd;
    uint32_t status;
    int chk;

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (node_ctx_p->protocol != 2)
        return -1;
    
    status = abcdk_option_get_int(stream_ctx_p->rsp_hdr,"Status",0,0);
    if(status == 0)
        return -3;
    
    /*记录活动时间。*/
    node_ctx_p->h2_send_active = _abcdk_https_clock(0);

    /*构造状态行。*/
    stream_ctx_p->h2_rsp_hdrs[stream_ctx_p->h2_rsp_count].name = (uint8_t*)":status";
    stream_ctx_p->h2_rsp_hdrs[stream_ctx_p->h2_rsp_count].namelen = 7;
    stream_ctx_p->h2_rsp_hdrs[stream_ctx_p->h2_rsp_count].value = (uint8_t*)abcdk_http_status_desc(status);
    stream_ctx_p->h2_rsp_hdrs[stream_ctx_p->h2_rsp_count].valuelen = 3; //只有数字，不能有字符。

    stream_ctx_p->h2_rsp_count += 1;

    /*遍历其它行。*/
    it.opaque = stream_ctx_p;
    it.dump_cb = _abcdk_https_rsp_hdr_dump_cb;
    abcdk_option_scan(stream_ctx_p->rsp_hdr,&it);

    data_prd.source.fd = -1;
    data_prd.read_callback = _abcdk_https_h2_response_read_cb;

    chk = nghttp2_submit_response(node_ctx_p->h2_handle, stream_ctx_p->id, stream_ctx_p->h2_rsp_hdrs, stream_ctx_p->h2_rsp_count, &data_prd);
    if(chk != 0)
        return -4;

#ifdef NGHTTP2_H
    /*恢复发送。非常重要。*/
    nghttp2_session_resume_data(node_ctx_p->h2_handle,stream_ctx_p->id);
#endif //NGHTTP2_H

    return 0;
#else  //NGHTTP2_H
    return -1;
#endif //NGHTTP2_H
}

static int _abcdk_https_response_header(abcdk_https_stream_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    int chk;

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (node_ctx_p->protocol == 1)
    {
        chk = _abcdk_https_response_header_h1(stream);
        if(chk != 0)
            return -1;
    }
    else if (node_ctx_p->protocol == 2)
    {
        chk = _abcdk_https_response_header_h2(stream);
        if(chk != 0)
            return -1;
    }

    abcdk_stcp_send_watch(stream_ctx_p->io_node);

    return 0;
}

static int _abcdk_https_response_body_h1(abcdk_https_stream_t *stream, abcdk_object_t *data)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    int chunked_body = 0;
    size_t chunked_size = 0;
    int chk;

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (node_ctx_p->protocol != 1)
        return -1;

    chunked_body = abcdk_option_exist(stream_ctx_p->rsp_hdr, "Transfer-Encoding");
    if (chunked_body)
    {
        /*发送分块头部。*/
        chunked_size = (data ? data->sizes[0] : 0);
        chk = abcdk_stcp_post_format(stream_ctx_p->io_node, 18, "%zx\r\n", chunked_size);
        if (chk != 0)
            return -1;

        /*可能是结束包。*/
        stream_ctx_p->rsp_end = (chunked_size <= 0);
    }
    else
    {
        /*可能是结束包。*/
        stream_ctx_p->rsp_end = (!data ? 1 : 0);
    }

    /*如果不是结束包，发送有效的数据块。*/
    if (!stream_ctx_p->rsp_end)
    {
        chk = abcdk_stcp_post(stream_ctx_p->io_node, data,1);
        if (chk != 0)
            return -1;
    }

    /*发送分块尾部。*/
    if (chunked_body)
    {
        chk = abcdk_stcp_post_buffer(stream_ctx_p->io_node,"\r\n",2);
        if (chk != 0)
            return -1;
    }

    return 0;
}

static int _abcdk_https_response_body_h2(abcdk_https_stream_t *stream, abcdk_object_t *data)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    int chunked_body = 0;
    int chk;

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (node_ctx_p->protocol != 2)
        return -1;

    /*可能是结束包。*/
    stream_ctx_p->rsp_end = (!data ? 1 : 0);

    /*结束包不需要发送。*/
    if(stream_ctx_p->rsp_end)
        return 0;
    
    /*记录活动时间。*/
    node_ctx_p->h2_send_active = _abcdk_https_clock(0);

    chk = abcdk_stream_write(stream_ctx_p->h2_out, data);
    if (chk != 0)
        return -2;

#ifdef NGHTTP2_H
    /*恢复发送。非常重要。*/
    nghttp2_session_resume_data(node_ctx_p->h2_handle,stream_ctx_p->id);
#endif //NGHTTP2_H

    return 0;
}

static int _abcdk_https_response_body(abcdk_https_stream_t *stream, abcdk_object_t *data)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    int chunked_body = 0;
    int chk;

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (node_ctx_p->protocol == 1)
    {
        chk = _abcdk_https_response_body_h1(stream,data);
        if(chk != 0)
            return -1;
    }
    else if (node_ctx_p->protocol == 2)
    {
        chk = _abcdk_https_response_body_h2(stream,data);
        if(chk != 0)
            return -2;
    }

    abcdk_stcp_send_watch(stream_ctx_p->io_node);

    return 0;
}

void abcdk_https_response_ready(abcdk_https_stream_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    void *old_userdata;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (node_ctx_p->protocol == 1)
    {
        /*nothing to do.*/
    }
    else if (node_ctx_p->protocol == 2)
    {
#ifdef NGHTTP2_H
        /*恢复发送。非常重要。*/
        nghttp2_session_resume_data(node_ctx_p->h2_handle,stream_ctx_p->id);
#endif //NGHTTP2_H
    }

    abcdk_stcp_send_watch(stream_ctx_p->io_node);
}

int abcdk_https_response_header_vset(abcdk_https_stream_t *stream,const char *key, const char *val, va_list ap)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    char buf[4000] = {0};
    int chk;
    
    assert(stream != NULL && key !=NULL && val != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    ABCDK_ASSERT(!stream_ctx_p->rsp_hdr_sent,"应答数据已经发送完成,不能修改。");

    if(!stream_ctx_p->rsp_hdr)
    {
        stream_ctx_p->rsp_hdr = abcdk_option_alloc("");
        if(!stream_ctx_p->rsp_hdr)
            return -1;
        
        /*添加默认的应答头部。*/
        abcdk_option_fset(stream_ctx_p->rsp_hdr,"Status", "200");
        abcdk_option_fset(stream_ctx_p->rsp_hdr,"Server","%s",node_ctx_p->cfg.name);
        abcdk_option_fset(stream_ctx_p->rsp_hdr,"Date","%s",abcdk_time_format_gmt(NULL, stream_ctx_p->loc_ctx));
        abcdk_option_fset(stream_ctx_p->rsp_hdr,"Access-Control-Allow-Origin","%s",(node_ctx_p->cfg.a_c_a_o?node_ctx_p->cfg.a_c_a_o:"*"));
        abcdk_option_fset(stream_ctx_p->rsp_hdr,"Expires","0");
        abcdk_option_fset(stream_ctx_p->rsp_hdr,"Cache-Control","no-cache");
        abcdk_option_fset(stream_ctx_p->rsp_hdr,"Content-Type","application/octet-stream");

        if(node_ctx_p->protocol == 1)
        {
            abcdk_option_fset(stream_ctx_p->rsp_hdr,"Connection","keep-alive");
        }

    }

    /*有些HTTP/2客户端兼容性不友好，这里要过滤一下。*/
    if (node_ctx_p->protocol == 2)
    {
        if (abcdk_strcmp(key, "Transfer-Encoding", 0) == 0)
            return 0;
        if (abcdk_strcmp(key, "Connection", 0) == 0)
            return 0;
    }

    vsnprintf(buf,4000,val,ap);

    /*删除旧的。*/
    abcdk_option_remove(stream_ctx_p->rsp_hdr,key);

    /*添加新的。*/
    chk = abcdk_option_set(stream_ctx_p->rsp_hdr,key,buf);
    if(chk != 0)
        return -2;

    return 0;
}

int abcdk_https_response_header_set(abcdk_https_stream_t *stream, const char *key, const char *val, ...)
{
    int chk;
    assert(stream != NULL && key != NULL && val != NULL);

    va_list ap;
    va_start(ap, val);
    chk = abcdk_https_response_header_vset(stream, key, val, ap);
    va_end(ap);

    return chk;
}

void abcdk_https_response_header_unset(abcdk_https_stream_t *stream,const char *key)
{
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    
    assert(stream != NULL && key !=NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    ABCDK_ASSERT(!stream_ctx_p->rsp_hdr_sent,"应答数据已经发送完成,不能修改。");

    if(!stream_ctx_p->rsp_hdr)
        return;

    abcdk_option_remove(stream_ctx_p->rsp_hdr,key);
}

int abcdk_https_response_header_end(abcdk_https_stream_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    uint32_t status;
    const char *upgrade_val;
    int chk;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    ABCDK_ASSERT(!stream_ctx_p->rsp_hdr_sent,"应答的头部已经结束。");
    ABCDK_ASSERT(stream_ctx_p->rsp_hdr,"还未设置应答的头部信息。");
    
    /*如果未设置应答长度，并且当前数据包不是末尾包，则添加分块传输标志。*/
    chk = abcdk_option_exist(stream_ctx_p->rsp_hdr,"Content-Length");
    if(!chk)
    {
        chk = abcdk_https_response_header_set(stream, "Transfer-Encoding", "chunked");
        if(chk != 0)
            return -2;
    }

    status = abcdk_option_get_int(stream_ctx_p->rsp_hdr,"Status",0,0);

    /*按需转换为隧道模式。*/
    if (stream_ctx_p->protocol == 1 && status == 200)
    {
        if (abcdk_strcmp(stream_ctx_p->method->pstrs[0], "CONNECT", 0) == 0)
            stream_ctx_p->protocol = 4;
        else
        {
            upgrade_val = abcdk_receiver_header_line_getenv(stream_ctx_p->updata, "Upgrade", ':');
            if (upgrade_val && abcdk_strcmp(stream_ctx_p->method->pstrs[0], "websocket", 0) == 0)
                stream_ctx_p->protocol = 4;
        }
    }

    stream_ctx_p->rsp_hdr_sent = 1;
    chk = _abcdk_https_response_header(stream);
    if (chk != 0)
        return -3;

    _abcdk_https_log(stream_p,status);

    return 0;
}

int abcdk_https_response(abcdk_https_stream_t *stream, abcdk_object_t *data)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    uint32_t status;
    const char *upgrade_val;
    int chk;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    /*如果头部未结束，先结束头部应答。*/
    if(!stream_ctx_p->rsp_hdr_sent)
    {
        chk = abcdk_https_response_header_end(stream);
        if(chk != 0)
            return chk;
    }

    chk = _abcdk_https_response_body(stream, data);
    if (chk != 0)
        return -4;

    return 0;
}

int abcdk_https_response_buffer(abcdk_https_stream_t *stream,const void *data, size_t size)
{
    abcdk_object_t *obj;
    int chk;

    assert(stream != NULL && data != NULL && size >0);

    obj = abcdk_object_copyfrom(data,size);
    if(!obj)
        return -1;

    chk = abcdk_https_response(stream,obj);
    if(chk == 0)
        return 0;

    abcdk_object_unref(&obj);
    return -2;
}

int abcdk_https_response_format(abcdk_https_stream_t *stream,int max, const char *fmt, ...)
{
    int chk;

    assert(stream != NULL && max >0 && fmt != NULL);
    
    va_list ap;
    va_start(ap, fmt);
    chk = abcdk_https_response_vformat(stream,max,fmt,ap);
    va_end(ap);

    return chk;
}

int abcdk_https_response_vformat(abcdk_https_stream_t *stream,int max, const char *fmt, va_list ap)
{
    abcdk_object_t *obj;
    int chk;

    assert(stream != NULL && max >0 && fmt != NULL);

    obj = abcdk_object_vprintf(max,fmt,ap);
    if(!obj)
        return -1;

    chk = abcdk_https_response(stream,obj);
    if(chk == 0)
        return 0;

    abcdk_object_unref(&obj);
    return -2;
}

static int _abcdk_https_load_auth(void *opaque, const char *user, char pawd[160])
{
    abcdk_https_node_t *node_ctx_p;
    char tmp[PATH_MAX] = {0};
    int chk;

    node_ctx_p = (abcdk_https_node_t *)opaque;

    abcdk_dirdir(tmp, node_ctx_p->cfg.auth_path);
    abcdk_dirdir(tmp, user);

    chk = abcdk_load(tmp, pawd, 160, 0);
    if (chk > 0)
        return 0;
    else if (chk < 0)
        return -1;

    return -2;
}

int abcdk_https_check_auth(abcdk_https_stream_t *stream)
{
    abcdk_https_node_t *node_ctx_p;
    abcdk_https_stream_internal_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    abcdk_option_t *auth_opt = NULL;
    const char *auth_p;
    int is_proxy = 0;
    int chk;

    assert(stream != NULL);

    stream_p = (abcdk_object_t *)stream;
    stream_ctx_p = (abcdk_https_stream_internal_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_https_node_t *)abcdk_stcp_get_userdata(stream_ctx_p->io_node);

    if (!node_ctx_p->cfg.auth_path)
        return 0;

    auth_p = abcdk_https_request_header_get(stream, "Authorization");
    if (!auth_p)
    {
        auth_p = abcdk_https_request_header_get(stream, "Proxy-Authorization");
        is_proxy = (auth_p!=NULL);
    }

    /*如果客户端没携带授权，则提示客户端提交授权。*/
    if (!auth_p)
        goto ERR;

    abcdk_http_parse_auth(&auth_opt,auth_p);
    abcdk_option_set(auth_opt,"http-method",stream_ctx_p->method->pstrs[0]);

    chk = abcdk_http_check_auth(auth_opt,_abcdk_https_load_auth,node_ctx_p);
    abcdk_option_free(&auth_opt);

    if(chk == 0)
        return 0;

ERR:

    abcdk_https_response_header_set(stream,"Status","%d",(is_proxy ? 407 : 401));

    if(is_proxy)
        abcdk_https_response_header_set(stream,"Proxy-Authenticate","Digest realm=\"%s\", charset=utf-8, nonce=\"%llu\"",node_ctx_p->cfg.realm,abcdk_rand(0,UINT64_MAX-1));
    else 
        abcdk_https_response_header_set(stream,"WWW-Authenticate","Digest realm=\"%s\", charset=utf-8, nonce=\"%llu\"",node_ctx_p->cfg.realm,abcdk_rand(0,UINT64_MAX-1));

 
    abcdk_https_response(stream,NULL);

    return -1;
}