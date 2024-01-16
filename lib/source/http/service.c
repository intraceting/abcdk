/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/http/service.h"

#ifdef HAVE_NGHTTP2
#include <nghttp2/nghttp2.h>
#endif // HAVE_NGHTTP2

#ifdef HAVE_LIBMAGIC
#include <magic.h>
#endif // HAVE_LIBMAGIC

/**简单的HTTP服务。*/
struct _abcdk_http_service
{
    /*配置。*/
    const abcdk_http_service_config_t *cfg;

    /*IO对象*/
    abcdk_asynctcp_t *io_ctx;

    /*监听。*/
    abcdk_asynctcp_node_t *io_listen_node;

    /*SSL监听。*/
    abcdk_asynctcp_node_t *io_listen_ssl_node;

    /*时间环境*/
    locale_t loc_ctx;

    /*文件类型探测器。*/
#ifdef _MAGIC_H
    struct magic_set *magic_ctx;
#endif // _MAGIC_H

    /*时间格式符。*/
    const char *timefmt;
    const char *timefmt_lc;

    /*追踪ID。*/
    volatile uint64_t tid_idx;

    /*随机数种子。*/
    uint64_t rand_seed;

}; // abcdk_http_service_t;

/**HTTP节点。*/
typedef struct _abcdk_http_service_node
{
    /*父级。*/
    abcdk_http_service_t *father;

    /*是否为服务端。*/
    int server;

    /*协议。1：HTTP/1.1/1.0/0.9 2: HTTP/2 3：HTTP/3 4：TUNNEL*/
    int protocol;

    /*远程地址。*/
    char remote_addr[NAME_MAX];

    /*SSL环境。*/
    SSL_CTX *ssl_ctx;

#ifdef NGHTTP2_H
    nghttp2_session_callbacks *h2_cbs;
    nghttp2_session *h2_handle;
#endif // NGHTTP2_H

    /*流容器。*/
    abcdk_map_t *stream_map;

    /*是否为SSL安全链路。*/
    int ssl_ok;

} abcdk_http_service_node_t;

/*流。*/
struct _abcdk_http_service_stream
{
    /*流ID。0 HTTP/1.1/1.0./0.9，> 0 HTTP/2。*/
    int id;

    /*追踪ID。*/
    uint64_t tid;

    /*协议。1：HTTP 4：TUNNEL*/
    int protocol;


    /*IO节点。*/
    abcdk_asynctcp_node_t *io_node;

    /*上行数据。*/
    abcdk_receiver_t *updata;

    abcdk_object_t *method;
    abcdk_object_t *script;
    abcdk_object_t *version;
    abcdk_object_t *host;
    abcdk_object_t *scheme;

    /*H2的头部是否已经结束。*/
    int h2_req_hdr_end;

    /*H2输出流。*/
    abcdk_stream_t *h2_out;
    
    /*应答是否结束。*/
    int rsp_end;

    /*用户环境指针。*/
    void *userdata;

}; // abcdk_http_service_stream_t;

static void _abcdk_http_service_stream_destructor_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_asynctcp_node_t *io_node_p;
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;

    io_node_p = (abcdk_asynctcp_node_t *)opaque;
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(io_node_p);
    stream_ctx_p = (abcdk_http_service_stream_t *)obj->pptrs[ABCDK_MAP_VALUE];

    /*通知连接已关闭。*/
    if(node_ctx_p->father->cfg->close_cb)
        node_ctx_p->father->cfg->close_cb(obj,stream_ctx_p->userdata);

    abcdk_receiver_unref(&stream_ctx_p->updata);
    abcdk_object_unref(&stream_ctx_p->method);
    abcdk_object_unref(&stream_ctx_p->script);
    abcdk_object_unref(&stream_ctx_p->version);
    abcdk_object_unref(&stream_ctx_p->host);
    abcdk_object_unref(&stream_ctx_p->scheme);
    abcdk_stream_destroy(&stream_ctx_p->h2_out);
    abcdk_asynctcp_unref(&stream_ctx_p->io_node);
}

static void _abcdk_http_service_stream_construct_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_asynctcp_node_t *io_node_p;
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    
    io_node_p = (abcdk_asynctcp_node_t *)opaque;
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(io_node_p);
    stream_ctx_p = (abcdk_http_service_stream_t *)obj->pptrs[ABCDK_MAP_VALUE];

    stream_ctx_p->id = *((int *)obj->pptrs[ABCDK_MAP_KEY]);
    stream_ctx_p->tid = abcdk_atomic_add_and_fetch(&node_ctx_p->father->tid_idx,1);
    stream_ctx_p->protocol = (node_ctx_p->protocol == 4 ? 4 : 1);//隧道不一样。
    stream_ctx_p->io_node = abcdk_asynctcp_refer(io_node_p);
    stream_ctx_p->h2_out = abcdk_stream_create();

    /*通知新连接到来。*/
    if(node_ctx_p->father->cfg->accept_cb)
        node_ctx_p->father->cfg->accept_cb(node_ctx_p->father->cfg->opaque, obj);
}

static void _abcdk_http_service_node_destroy_cb(void *userdata)
{
    abcdk_http_service_node_t *ctx;

    if (!userdata)
        return;

    ctx = (abcdk_http_service_node_t *)userdata;

    abcdk_openssl_ssl_ctx_free(&ctx->ssl_ctx);

#ifdef NGHTTP2_H
    if (ctx->h2_handle)
        nghttp2_session_del(ctx->h2_handle);
    if (ctx->h2_cbs)
        nghttp2_session_callbacks_del(ctx->h2_cbs);
#endif // #NGHTTP2_H

    abcdk_map_destroy(&ctx->stream_map);
}

static abcdk_asynctcp_node_t *_abcdk_http_service_node_alloc(abcdk_http_service_t *father, int server, int ssl)
{
    abcdk_asynctcp_node_t *node;
    abcdk_http_service_node_t *node_ctx;

    node = abcdk_asynctcp_alloc(father->io_ctx, sizeof(abcdk_http_service_node_t), _abcdk_http_service_node_destroy_cb);
    if (!node)
        return NULL;

    node_ctx = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    node_ctx->father = father;
    node_ctx->server = server;
    node_ctx->protocol = 1;

    if (ssl)
    {
        /*可能无证书。*/
        if (father->cfg->cert_file && father->cfg->key_file)
            node_ctx->ssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(server, father->cfg->ca_file, father->cfg->ca_path, father->cfg->cert_file, father->cfg->key_file, NULL);
    }

    return node;
}

static void _abcdk_http_service_log(abcdk_object_t *stream, uint32_t status)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    char new_tname[18] = {0}, old_tname[18] = {0};
    const char *user_agent_p,*refer_p;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    /*只记录HTTP日志。*/
    if (node_ctx_p->protocol != 1 && node_ctx_p->protocol != 2)
        return;

    snprintf(new_tname, 16, "%x", stream_ctx_p->tid);

    pthread_getname_np(pthread_self(), old_tname, 18);
    pthread_setname_np(pthread_self(), new_tname);

    if (status)
    {
        abcdk_trace_output(LOG_INFO, "'%s'\n",abcdk_http_status_desc(status));
    }
    else
    {
        user_agent_p = abcdk_receiver_header_line_getenv(stream_ctx_p->updata,"User-Agent",':');
        refer_p = abcdk_receiver_header_line_getenv(stream_ctx_p->updata,"Referer",':');

        abcdk_trace_output(LOG_INFO, "'%s' '%s' '%s' '%s' '%s'\n",
                           node_ctx_p->remote_addr, 
                           stream_ctx_p->method->pstrs[0], 
                           stream_ctx_p->script->pstrs[0],
                           (refer_p?refer_p:"-"), 
                           (user_agent_p?user_agent_p:"-"));
    }

    pthread_setname_np(pthread_self(), old_tname);
}


static void _abcdk_http_service_process(abcdk_object_t *stream)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    const char *upgrade_val;
    int chk;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    _abcdk_http_service_log(stream,0);

    /*通知应用层数据到达。*/
    node_ctx_p->father->cfg->request_cb(node_ctx_p->father->cfg->opaque,stream);

    /*按需转换为隧道模式。*/
    if (stream_ctx_p->protocol == 1)
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
}

#ifdef NGHTTP2_H

static void _abcdk_http_service_process_h2(abcdk_object_t *stream)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    if (stream_ctx_p->protocol == 1)
    {
        ;
    }
    else if (stream_ctx_p->protocol == 4)
    {
        ;
    }

    _abcdk_http_service_process(stream);
}

static int _abcdk_http_service_h2_frame_recv_cb(nghttp2_session *session, const nghttp2_frame *frame, void *user_data)
{
    abcdk_asynctcp_node_t *node = (abcdk_asynctcp_node_t *)user_data;
    abcdk_http_service_node_t *node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);
    abcdk_http_service_stream_t *stream_ctx_p;
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

    stream_ctx_p = (abcdk_http_service_stream_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

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

    _abcdk_http_service_process_h2(stream_p);



    /*一定要回收。*/
    abcdk_receiver_unref(&stream_ctx_p->updata);

    return 0;
}

static int _abcdk_http_service_h2_data_chunk_recv_cb(nghttp2_session *session, uint8_t flags, int32_t stream_id, const uint8_t *data, size_t len, void *user_data)
{
    abcdk_asynctcp_node_t *node = (abcdk_asynctcp_node_t *)user_data;
    abcdk_http_service_node_t *node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);
    abcdk_http_service_stream_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    size_t remain = 0;
    int chk;

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, 0);
    if (!stream_p)
        return -1;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

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

static int _abcdk_http_service_h2_header_cb(nghttp2_session *session, const nghttp2_frame *frame,
                                            const uint8_t *name, size_t namelen,
                                            const uint8_t *value, size_t valuelen,
                                            uint8_t flags, void *user_data)
{
    abcdk_asynctcp_node_t *node = (abcdk_asynctcp_node_t *)user_data;
    abcdk_http_service_node_t *node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);
    abcdk_http_service_stream_t *stream_ctx_p;
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

    stream_ctx_p = (abcdk_http_service_stream_t *)stream_p->pptrs[ABCDK_MAP_VALUE];
  
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
      //  abcdk_trace_output(LOG_INFO, "%s: {%s}{%d}{%d}\r\n", name, value,(frame->hd.flags),(flags));

        abcdk_receiver_append(stream_ctx_p->updata, name, namelen, &remain);
        abcdk_receiver_append(stream_ctx_p->updata, ": ", 2, &remain);
        abcdk_receiver_append(stream_ctx_p->updata, value, valuelen, &remain);
        abcdk_receiver_append(stream_ctx_p->updata, "\r\n", 2, &remain);
        if (stream_ctx_p->h2_req_hdr_end = (flags & NGHTTP2_FLAG_END_HEADERS))
            abcdk_receiver_append(stream_ctx_p->updata, "\r\n", 2, &remain);
    }

    return 0;
}

static int _abcdk_http_service_h2_begin_headers_cb(nghttp2_session *session, const nghttp2_frame *frame, void *user_data)
{
    abcdk_asynctcp_node_t *node = (abcdk_asynctcp_node_t *)user_data;
    abcdk_http_service_node_t *node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);
    abcdk_http_service_stream_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    int stream_id;

    stream_id = frame->hd.stream_id;

    /*过滤掉不需要的，但也不能返回失败。*/
    if (frame->hd.type != NGHTTP2_HEADERS || frame->headers.cat != NGHTTP2_HCAT_REQUEST)
        return 0;

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, sizeof(abcdk_http_service_stream_t));
    if (!stream_p)
        return -1;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    if (!stream_ctx_p->updata)
    {
        if(stream_ctx_p->protocol == 1)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_HTTP, node_ctx_p->father->cfg->up_max_size, node_ctx_p->father->cfg->up_tmp_path);
        else if(stream_ctx_p->protocol == 4)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM, node_ctx_p->father->cfg->up_max_size, node_ctx_p->father->cfg->up_tmp_path);
    }    

    if (!stream_ctx_p->updata)
        return -2;

    return 0;
}

static int _abcdk_http_service_h2_stream_close_cb(nghttp2_session *session, int32_t stream_id, uint32_t error_code, void *user_data)
{
    abcdk_asynctcp_node_t *node = (abcdk_asynctcp_node_t *)user_data;
    abcdk_http_service_node_t *node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    /*移除这个节点。*/
    abcdk_map_remove2(node_ctx_p->stream_map, &stream_id);

    return 0;
}

static ssize_t _abcdk_http_service_h2_send_cb(nghttp2_session *session, const uint8_t *data, size_t length, int flags, void *user_data)
{
    abcdk_asynctcp_node_t *node = (abcdk_asynctcp_node_t *)user_data;
    abcdk_http_service_node_t *node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);
    int chk;

    chk = abcdk_asynctcp_post_buffer(node, data, length);
    if (chk != 0)
        return 0;

    return length;
}

static ssize_t _abcdk_http_service_h2_response_read_cb(nghttp2_session *session, int32_t stream_id, uint8_t *buf, size_t length,
                                                       uint32_t *data_flags, nghttp2_data_source *source, void *user_data)
{
    abcdk_asynctcp_node_t *node = (abcdk_asynctcp_node_t *)user_data;
    abcdk_http_service_node_t *node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);
    abcdk_http_service_stream_t *stream_ctx_p;
    abcdk_object_t *stream_p;
    ssize_t rlen;

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, 0);
    if (!stream_p)
        return -1;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    /*通知链路空闲。*/
    if(node_ctx_p->father->cfg->output_cb)
        node_ctx_p->father->cfg->output_cb(node_ctx_p->father->cfg->opaque,stream_p);

    rlen = abcdk_stream_read(stream_ctx_p->h2_out,buf,length);

    if(rlen <= 0 && stream_ctx_p->rsp_end)
        *data_flags |= NGHTTP2_DATA_FLAG_EOF;

    return rlen;
}

#endif // NGHTTP2_H


static void _abcdk_http_service_process_h1(abcdk_object_t *stream)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    const char *line_p;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

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

        if (node_ctx_p->ssl_ok)
            stream_ctx_p->scheme = abcdk_object_copyfrom("https", 5);
        else
            stream_ctx_p->scheme = abcdk_object_copyfrom("http", 4);
    }
    else if (stream_ctx_p->protocol == 4)
    {

    }

    _abcdk_http_service_process(stream);

ERR:

    abcdk_asynctcp_set_timeout(stream_ctx_p->io_node,1);

}

static void _abcdk_http_service_request_h1(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_object_t *stream_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    int stream_id = 0;
    const char *upgrade_val;
    int chk;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, sizeof(abcdk_http_service_stream_t));
    if (!stream_p)
        goto ERR;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    if (!stream_ctx_p->updata)
    {
        if (stream_ctx_p->protocol == 1)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_HTTP, node_ctx_p->father->cfg->up_max_size, node_ctx_p->father->cfg->up_tmp_path);
        else if (stream_ctx_p->protocol == 4)
            stream_ctx_p->updata = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_STREAM, node_ctx_p->father->cfg->up_max_size, node_ctx_p->father->cfg->up_tmp_path);
        else
            goto ERR;
    }

    if (!stream_ctx_p->updata)
        goto ERR;

    chk = abcdk_receiver_append(stream_ctx_p->updata, data, size, remain);
    if (chk < 0)
        goto ERR;
    else if (chk == 0) /*数据包不完整，继续接收。*/
        return;

    _abcdk_http_service_process_h1(stream_p);
    
    /*一定要回收。*/
    abcdk_receiver_unref(&stream_ctx_p->updata);

    /*No Error.*/
    return;

ERR:

    abcdk_asynctcp_set_timeout(node, 1);
}

static void _abcdk_http_service_request_h2(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_http_service_node_t *node_ctx_p;
    ssize_t chk;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

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

    abcdk_asynctcp_set_timeout(node, 1);
}

static void _abcdk_http_service_output_h1(abcdk_asynctcp_node_t *node)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_object_t *stream_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    int stream_id = 0;
    int chk;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    stream_p = abcdk_map_find2(node_ctx_p->stream_map, &stream_id, 0);
    if (!stream_p)
        return;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream_p->pptrs[ABCDK_MAP_VALUE];

    /*通知链路空闲。*/
    if(node_ctx_p->father->cfg->output_cb)
        node_ctx_p->father->cfg->output_cb(node_ctx_p->father->cfg->opaque,stream_p);
}

static void _abcdk_http_service_output_h2(abcdk_asynctcp_node_t *node)
{
    abcdk_http_service_node_t *node_ctx_p;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

#ifdef NGHTTP2_H
    /*把缓存数据串行化，并通过回调发送出去。*/
    nghttp2_session_send(node_ctx_p->h2_handle);
#endif //NGHTTP2_H
}

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
static int _abcdk_http_service_alpn_select_cb(SSL *ssl, const unsigned char **out, unsigned char *outlen,
                                              const unsigned char *in, unsigned int inlen, void *arg)
{
    unsigned char *srv;
    unsigned int srvlen;
    size_t alpn_flag;

    alpn_flag = (size_t)arg;

    /*协议选择时，仅做指针的复制，因此这里要么用静态的变量，要么创建一个全局有效的。*/
    static unsigned char srv1[] = {"\x08http/1.1\x06tunnel"};
    static unsigned char srv2[] = {"\x02h2\x08http/1.1\x06tunnel"};
    static unsigned char srv3[] = {"\x06tunnel"};

    if (alpn_flag == 3)
    {
        srv = srv3;
        srvlen = sizeof(srv3) - 1;
    }
    else if (alpn_flag == 2)
    {
        srv = srv2;
        srvlen = sizeof(srv2) - 1;
    }
    else
    {
        srv = srv1;
        srvlen = sizeof(srv1) - 1;
    }

    /*服务端在客户端支持的协议列表中选择一个支持协议，从左到右按顺序匹配。*/
    if (SSL_select_next_proto((unsigned char **)out, outlen, in, inlen, srv, srvlen) != OPENSSL_NPN_NEGOTIATED)
    {
        return SSL_TLSEXT_ERR_ALERT_FATAL;
    }

    return SSL_TLSEXT_ERR_OK;
}
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H

static void _abcdk_http_service_set_alpn(abcdk_asynctcp_node_t *node, size_t alpn_flag)
{
    abcdk_http_service_node_t *node_ctx_p;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
    SSL_CTX_set_alpn_select_cb(node_ctx_p->ssl_ctx, _abcdk_http_service_alpn_select_cb, (void *)alpn_flag);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H
}

static void _abcdk_http_service_prepare_cb(abcdk_asynctcp_node_t **node, abcdk_asynctcp_node_t *listen)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_http_service_node_t *listen_ctx_p, *node_ctx_p;
    abcdk_object_t *userdata_p = NULL;

    listen_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(listen);

    node_p = _abcdk_http_service_node_alloc(listen_ctx_p->father, 1, 0);
    if (!node_p)
        return;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->stream_map = abcdk_map_create(10);
    node_ctx_p->stream_map->construct_cb = _abcdk_http_service_stream_construct_cb;
    node_ctx_p->stream_map->destructor_cb = _abcdk_http_service_stream_destructor_cb;
    node_ctx_p->stream_map->opaque = node_p;

#ifdef NGHTTP2_H
    nghttp2_session_callbacks_new(&node_ctx_p->h2_cbs);
    nghttp2_session_callbacks_set_on_frame_recv_callback(node_ctx_p->h2_cbs, _abcdk_http_service_h2_frame_recv_cb);
    nghttp2_session_callbacks_set_on_data_chunk_recv_callback(node_ctx_p->h2_cbs, _abcdk_http_service_h2_data_chunk_recv_cb);
    nghttp2_session_callbacks_set_on_header_callback(node_ctx_p->h2_cbs, _abcdk_http_service_h2_header_cb);
    nghttp2_session_callbacks_set_on_begin_headers_callback(node_ctx_p->h2_cbs, _abcdk_http_service_h2_begin_headers_cb);
    nghttp2_session_callbacks_set_on_stream_close_callback(node_ctx_p->h2_cbs, _abcdk_http_service_h2_stream_close_cb);
    nghttp2_session_callbacks_set_send_callback(node_ctx_p->h2_cbs, _abcdk_http_service_h2_send_cb);

    nghttp2_session_server_new(&node_ctx_p->h2_handle, node_ctx_p->h2_cbs, node_p);

    nghttp2_settings_entry iv[] = {
        {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS, 100},
        {NGHTTP2_SETTINGS_MAX_HEADER_LIST_SIZE, 100},
        {NGHTTP2_SETTINGS_MAX_FRAME_SIZE, 65535}};

    /*必须要设置。*/
    nghttp2_submit_settings(node_ctx_p->h2_handle, NGHTTP2_FLAG_NONE, iv, 3);
#endif // NGHTTP2_H

    /*准备完毕，返回。*/
    *node = node_p;
}

static void _abcdk_http_service_event_accept(abcdk_asynctcp_node_t *node, int *result)
{
    abcdk_http_service_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    /*默认：允许。*/
    *result = 0;

    if(node_ctx_p->father->cfg->firewall_cb)
        *result = node_ctx_p->father->cfg->firewall_cb(node_ctx_p->father->cfg->opaque, node_ctx_p->remote_addr);
    
    if(*result != 0)
        abcdk_trace_output(LOG_INFO, "禁止客户端('%s')连接到本机。", node_ctx_p->remote_addr);
}

static void _abcdk_http_service_event_connect(abcdk_asynctcp_node_t *node)
{
    abcdk_http_service_node_t *node_ctx_p;
    SSL *ssl_p;
    char ptl_sel_name[256] = {0};
    int chk;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    /*默认协议。*/
    node_ctx_p->protocol = 1;

    /*设置超时。*/
    if (node_ctx_p->father->cfg->stimeout > 0)
        abcdk_asynctcp_set_timeout(node, node_ctx_p->father->cfg->stimeout * 1000);
    else
        abcdk_asynctcp_set_timeout(node, 525600 * 1000);

    if (!node_ctx_p->server)
        abcdk_asynctcp_get_sockaddr_str(node, NULL, node_ctx_p->remote_addr);

    ssl_p = abcdk_asynctcp_ssl(node);
    if (!ssl_p)
        goto END;

#ifdef HEADER_SSL_H

    /*检查SSL验证结果。*/
    chk = SSL_get_verify_result(ssl_p);
    if (chk != X509_V_OK)
    {
        abcdk_trace_output(LOG_INFO, "验证('%s')的证书失败，证书已过期或未生效。", node_ctx_p->remote_addr);

        /*修改超时，使用超时检测器关闭。*/
        abcdk_asynctcp_set_timeout(node, 1);
        return;
    }

    /*安全链路。*/
    node_ctx_p->ssl_ok = 1;

    chk = abcdk_openssl_ssl_get_alpn_selected(ssl_p, ptl_sel_name);
    if (chk == 0)
    {
        if (abcdk_strcmp("tunnel", ptl_sel_name, 0) == 0)
            node_ctx_p->protocol = 4;
        else if (abcdk_strncmp("http", ptl_sel_name, 4, 0) == 0)
            node_ctx_p->protocol = 1;
        else if (abcdk_strcmp("h2", ptl_sel_name, 0) == 0)
            node_ctx_p->protocol = 2;
    }

#endif // HEADER_SSL_H

END:

    abcdk_trace_output(LOG_INFO, "本机与%s('%s')的连接已经建立。", (node_ctx_p->server ? "客户端" : "服务端"), node_ctx_p->remote_addr);

    /*已连接到远端，注册读写事件。*/
    abcdk_asynctcp_recv_watch(node);
    abcdk_asynctcp_send_watch(node);
}

static void _abcdk_http_service_event_output(abcdk_asynctcp_node_t *node)
{
    abcdk_http_service_node_t *node_ctx_p;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    if(node_ctx_p->protocol == 1)
    {
        _abcdk_http_service_output_h1(node);
    }
    else if( node_ctx_p->protocol == 2)
    {
        _abcdk_http_service_output_h2(node);

    }
    else if( node_ctx_p->protocol == 4)
    {

    }

}

static void _abcdk_http_service_event_close(abcdk_asynctcp_node_t *node)
{
    abcdk_http_service_node_t *node_ctx_p;
    char ptl_sel_name[256] = {0};
    int chk;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    /*一定要在这里释放，否则在单路复用时，由于多次引用的原因会使当前链路得不到释放。*/
    abcdk_map_destroy(&node_ctx_p->stream_map);

    abcdk_trace_output(LOG_INFO, "本机与%s('%s')的连接已经断开。", (node_ctx_p->server ? "客户端" : "服务端"), node_ctx_p->remote_addr);
}

static void _abcdk_http_service_event_cb(abcdk_asynctcp_node_t *node, uint32_t event, int *result)
{
    abcdk_http_service_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    if (event == ABCDK_ASYNCTCP_EVENT_ACCEPT)
    {
        _abcdk_http_service_event_accept(node,result);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CONNECT)
    {
        _abcdk_http_service_event_connect(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_INPUT)
    {
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_OUTPUT)
    {
        _abcdk_http_service_event_output(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CLOSE || event == ABCDK_ASYNCTCP_EVENT_INTERRUPT)
    {
        _abcdk_http_service_event_close(node);
    }
}


static void _abcdk_http_service_request_cb(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_http_service_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    if (node_ctx_p->protocol == 1)
    {
        _abcdk_http_service_request_h1(node, data, size, remain);
    }
    else if (node_ctx_p->protocol == 2)
    {
        _abcdk_http_service_request_h2(node, data, size, remain);
    }
    else if (node_ctx_p->protocol == 4)
    {
    }
}

static int _abcdk_http_service_start_listen(abcdk_http_service_t *ctx, int ssl)
{
    abcdk_sockaddr_t addr = {0};
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_asynctcp_node_t *node_p = NULL;
    SSL_CTX *ssl_ctx_p = NULL;
    abcdk_asynctcp_callback_t cb = {0};
    int chk;

    if (ssl)
    {
        if (!ctx->cfg->listen_ssl)
        {
            abcdk_trace_output(LOG_ERR, "SSL监听地址不存在，忽略。");
            return 0;
        }

        if (!ctx->cfg->cert_file || access(ctx->cfg->cert_file, R_OK) != 0)
        {
            abcdk_trace_output(LOG_ERR, "服务端证书不存在或无法访问('%s')，无法启动SSL监听。", (ctx->cfg->cert_file ? ctx->cfg->cert_file : ""));
            return -22;
        }

        if (!ctx->cfg->key_file || access(ctx->cfg->key_file, R_OK) != 0)
        {
            abcdk_trace_output(LOG_ERR, "服务端私钥不存在或无法访问('%s')，无法启动SSL监听。", (ctx->cfg->key_file ? ctx->cfg->key_file : ""));
            return -22;
        }

        chk = abcdk_sockaddr_from_string(&addr, ctx->cfg->listen_ssl, 0);
        if (chk != 0)
        {
            abcdk_trace_output(LOG_ERR, "无法解析地址('%s')。", ctx->cfg->listen_ssl);
            return -1;
        }

        ctx->io_listen_ssl_node = node_p = _abcdk_http_service_node_alloc(ctx, 1, 1);
        if (!ctx->io_listen_ssl_node)
            return -2;

        node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(ctx->io_listen_ssl_node);

        ssl_ctx_p = node_ctx_p->ssl_ctx;
        if (!ssl_ctx_p)
            return -3;

#ifdef NGHTTP2_H
        if (ctx->cfg->enable_h2)
            _abcdk_http_service_set_alpn(ctx->io_listen_ssl_node, 2);
        else
#endif // NGHTTP2_H
            _abcdk_http_service_set_alpn(ctx->io_listen_ssl_node, 1);
    }
    else
    {
        chk = abcdk_sockaddr_from_string(&addr, ctx->cfg->listen, 0);
        if (chk != 0)
        {
            abcdk_trace_output(LOG_ERR, "无法解析地址('%s')。", ctx->cfg->listen);
            return -1;
        }

        ctx->io_listen_node = node_p = _abcdk_http_service_node_alloc(ctx, 1, 0);
        if (!ctx->io_listen_node)
            return -3;
    }

    cb.prepare_cb = _abcdk_http_service_prepare_cb;
    cb.event_cb = _abcdk_http_service_event_cb;
    cb.request_cb = _abcdk_http_service_request_cb;
    chk = abcdk_asynctcp_listen(node_p, ssl_ctx_p, &addr, &cb);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "启动监听失败，无权限或端口被占用。");
        return -4;
    }

    return 0;
}

void abcdk_http_service_destroy(abcdk_http_service_t **ctx)
{
    abcdk_http_service_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

#ifdef _MAGIC_H
    if (ctx_p->magic_ctx)
        magic_close(ctx_p->magic_ctx);
#endif // _MAGIC_H

    if(ctx_p->loc_ctx)
        freelocale(ctx_p->loc_ctx);

    abcdk_asynctcp_stop(&ctx_p->io_ctx);
    abcdk_asynctcp_unref(&ctx_p->io_listen_node);
    abcdk_asynctcp_unref(&ctx_p->io_listen_ssl_node);
    abcdk_heap_free(ctx_p);
}

abcdk_http_service_t *abcdk_http_service_create(const abcdk_http_service_config_t *cfg)
{
    abcdk_http_service_t *ctx;
    int chk;

    assert(cfg != NULL);
    assert(cfg->request_cb != NULL);

    ctx = (abcdk_http_service_t *)abcdk_heap_alloc(sizeof(abcdk_http_service_t));
    if (!ctx)
        return NULL;

    /*copy*/
    ctx->cfg = cfg;

    ctx->io_ctx = abcdk_asynctcp_start(ctx->cfg->max_client, -1);
    if (!ctx->io_ctx)
        goto ERR;

    chk = _abcdk_http_service_start_listen(ctx, 0);
    if (chk != 0)
        goto ERR;

    chk = _abcdk_http_service_start_listen(ctx, 1);
    if (chk != 0)
        goto ERR;

    ctx->loc_ctx = newlocale(LC_ALL_MASK,"en_US.UTF-8",NULL);

#ifdef _MAGIC_H
    ctx->magic_ctx = magic_open(MAGIC_MIME | MAGIC_SYMLINK);
    if (ctx->magic_ctx)
        magic_load(ctx->magic_ctx, NULL);
#endif // _MAGIC_H

    ctx->timefmt = "%a, %d %b %Y %H:%M:%S GMT";
    ctx->timefmt_lc = "%Y-%m-%d %H:%M:%S";
    ctx->tid_idx = 0;
    ctx->rand_seed = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,9);

    return ctx;
ERR:

    abcdk_http_service_destroy(&ctx);

    return NULL;
}

void *abcdk_http_service_get_userdata(abcdk_object_t *stream)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;

    assert(stream != NULL);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    return stream_ctx_p->userdata;
}

void *abcdk_http_service_set_userdata(abcdk_object_t *stream,void *userdata)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    void *old_userdata;

    assert(stream != NULL);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    old_userdata = stream_ctx_p->userdata;
    stream_ctx_p->userdata = userdata;
    
    return old_userdata;

}

const char *abcdk_http_service_address_remote(abcdk_object_t *stream)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;

    assert(stream != NULL);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    return node_ctx_p->remote_addr;
}

const char *abcdk_http_service_request_header_get(abcdk_object_t *stream, const char *key)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;

    assert(stream != NULL);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

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

const char* abcdk_http_service_request_header_getline(abcdk_object_t *stream,int line)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;

    assert(stream != NULL && line >= 1 && line < 100);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    /*HTTP协议有头部数据。*/
    if (stream_ctx_p->protocol != 1)
        goto ERR;
    
    return abcdk_receiver_header_line(stream_ctx_p->updata, line);
ERR:

    return NULL;
}

const char *abcdk_http_service_request_body_get(abcdk_object_t *stream, size_t *len)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    const void *body_p;
    size_t body_l;

    assert(stream != NULL);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    body_l = abcdk_receiver_body_length(stream_ctx_p->updata);
    if (body_l <= 0)
        return NULL;

    if (len)
        *len = body_l;

    body_p = abcdk_receiver_body(stream_ctx_p->updata, 0);

    return body_p;
}

static int _abcdk_http_service_response_header_h1(abcdk_object_t *stream, uint32_t status,abcdk_object_t *data)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    int chk;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    chk = abcdk_asynctcp_post_format(stream_ctx_p->io_node,1000,"HTTP/1.1 %s\r\n",abcdk_http_status_desc(status));
    if (chk != 0)
        return -1;

    chk = abcdk_asynctcp_post(stream_ctx_p->io_node, data);
    if (chk != 0)
        return -2;

    chk = abcdk_asynctcp_post_buffer(stream_ctx_p->io_node, "\r\n",2);
    if (chk != 0)
        return -2;

    return 0;
}

static int _abcdk_http_service_response_header_h2(abcdk_object_t *stream, uint32_t status,abcdk_object_t *data)
{
#ifdef NGHTTP2_H
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    nghttp2_nv hdrs[100] = {0};
    nghttp2_data_provider data_prd;
    const char *p, *p_next;
    int chk, i = 0;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    hdrs[i].name = (uint8_t*)":status";
    hdrs[i].namelen = strlen(hdrs[i].name);
    hdrs[i].value = (uint8_t*)abcdk_http_status_desc(status);
    hdrs[i].valuelen = 3; //只有数字，不能有字符。

    p = NULL;
    p_next = data->pstrs[0];

    for (i = 1; i < 100; i++)
    {
        p = abcdk_strtok(&p_next, ": ");
        if (!p)
            break;

        hdrs[i].name = (uint8_t *)p;
        hdrs[i].namelen = p_next - p;

        p_next += 2;
        p = abcdk_strtok(&p_next, "\r\n");
        if (!p)
            return -2;

        hdrs[i].value = (uint8_t *)p;
        hdrs[i].valuelen = p_next - p;

        p_next += 2;
    }

    data_prd.source.fd = -1;
    data_prd.read_callback = _abcdk_http_service_h2_response_read_cb;

    chk = nghttp2_submit_response(node_ctx_p->h2_handle, stream_ctx_p->id, hdrs, i, &data_prd);
    if(chk != 0)
        return -3;

    return 0;
#else  //NGHTTP2_H
    return -1;
#endif //NGHTTP2_H
}

static int _abcdk_http_service_response_header(abcdk_object_t *stream,uint32_t status,abcdk_object_t *data)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    int chk;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    if (node_ctx_p->protocol == 1)
    {
        chk = _abcdk_http_service_response_header_h1(stream,status,data);
        if(chk != 0)
            return -1;
    }
    else if (node_ctx_p->protocol == 2)
    {
        chk = _abcdk_http_service_response_header_h2(stream,status,data);
        if(chk != 0)
            return -1;
    }

    _abcdk_http_service_log(stream,status);

    /*激活发送。*/
    abcdk_asynctcp_send_watch(stream_ctx_p->io_node);

    return 0;
}

static int _abcdk_http_service_response_body(abcdk_object_t *stream, abcdk_object_t *data)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    int chk;

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    if(!data)
    {
        stream_ctx_p->rsp_end = 1;
        goto END;
    }

    if (node_ctx_p->protocol == 1)
    {
        chk = abcdk_asynctcp_post(stream_ctx_p->io_node,data);
        if(chk != 0)
            return -1;
    }
    else if (node_ctx_p->protocol == 2)
    {
        chk = abcdk_stream_write(stream_ctx_p->h2_out, data);
        if(chk !=0 )
            return -2;
    }

END:

    abcdk_asynctcp_send_watch(stream_ctx_p->io_node);

    return 0;
}

int abcdk_http_service_response_vheader(abcdk_object_t *stream,uint32_t status,int max, const char *fmt, va_list ap)
{
    abcdk_object_t *obj;
    int chk;

    assert(stream != NULL && status >=100 && max > 0 && fmt != NULL);

    obj = abcdk_object_vprintf(max,fmt,ap);
    if(!obj)
        return -1;

    chk = _abcdk_http_service_response_header(stream,status,obj);
    if(chk == 0)
        return 0;

    /*删除应答失败的。*/
    abcdk_object_unref(&obj);
    return chk;
}

int abcdk_http_service_response_header(abcdk_object_t *stream,uint32_t status,int max, const char *fmt, ...)
{
    int chk;

    assert(stream != NULL && status >=100 && max > 0 && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    chk = abcdk_http_service_response_vheader(stream,status,max, fmt, ap);
    va_end(ap);

    return chk;
}

int abcdk_http_service_response_body(abcdk_object_t *stream,abcdk_object_t *data)
{
    int chk;

    assert(stream != NULL);

    chk = _abcdk_http_service_response_body(stream, data);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_http_service_response_body_buffer(abcdk_object_t *stream, const void *data, size_t size)
{
    abcdk_object_t *obj;
    int chk;

    assert(stream != NULL && data != NULL && size > 0);

    obj = abcdk_object_copyfrom(data, size);
    if (!obj)
        return -1;

    chk = _abcdk_http_service_response_body(stream, obj);
    if (chk == 0)
        return 0;

    /*删除应答失败的。*/
    abcdk_object_unref(&obj);
    return -1;
}

int abcdk_http_service_response_nobody(abcdk_object_t *stream, uint32_t status, const char *a_c_a_m, const char *a_c_a_o)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    int chk;

    assert(stream != NULL && status >= 100);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    abcdk_http_service_response_header(stream, status, 300,
                                       "Server: %s\r\n"
                                       "Data: %s\r\n"
                                       "Access-Control-Allow-Origin: %s\r\n"
                                       "Access-Control-Allow-Methods: %s\r\n"
                                       "Content-Length: 0\r\n"
                                       "Cache-Control: no-cache\r\n"
                                       "Expires: 0\r\n",
                                       node_ctx_p->father->cfg->server_name,
                                       abcdk_time_format(node_ctx_p->father->timefmt, NULL, node_ctx_p->father->loc_ctx),
                                       (a_c_a_o && *a_c_a_o ? a_c_a_o : "*"),
                                       (a_c_a_m && *a_c_a_m ? a_c_a_m : "*"));

    abcdk_http_service_response_body(stream, NULL);

    return 0;
}

int abcdk_http_service_response(abcdk_object_t *stream, uint32_t status, abcdk_object_t *data, const char *type, const char *a_c_a_o)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    int chk;

    assert(stream != NULL && status >= 100 && data != NULL);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

#ifdef _MAGIC_H
    if (node_ctx_p->father->magic_ctx && type == NULL)
        type = magic_buffer(node_ctx_p->father->magic_ctx, data->pptrs[0],data->sizes[0]);
#endif //_MAGIC_H

    /*走到这里如果未确定类型，统一按二进制处理。*/
    if(!type)
        type = "application/octet-stream";

    abcdk_http_service_response_header(stream, status, 300,
                                       "Server: %s\r\n"
                                       "Data: %s\r\n"
                                       "Access-Control-Allow-Origin: %s\r\n"
                                       "Content-Length: %zd\r\n"
                                       "Content-Type: %s\r\n"
                                       "Cache-Control: no-cache\r\n"
                                       "Expires: 0\r\n",
                                       node_ctx_p->father->cfg->server_name,
                                       abcdk_time_format(node_ctx_p->father->timefmt, NULL, node_ctx_p->father->loc_ctx),
                                       (a_c_a_o && *a_c_a_o ? a_c_a_o : "*"),
                                       data->sizes[0],
                                       type);

    abcdk_http_service_response_body(stream, data);
    abcdk_http_service_response_body(stream, NULL);

    return 0;
}

int abcdk_http_service_response_buffer(abcdk_object_t *stream, uint32_t status, const char *data, size_t size, const char *type, const char *a_c_a_o)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    int chk;

    assert(stream != NULL && status >= 100 && data != NULL && size > 0);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

#ifdef _MAGIC_H
    if (node_ctx_p->father->magic_ctx && type == NULL)
        type = magic_buffer(node_ctx_p->father->magic_ctx, data, size);
#endif //_MAGIC_H

    /*走到这里如果未确定类型，统一按二进制处理。*/
    if(!type)
        type = "application/octet-stream";

    abcdk_http_service_response_header(stream, status, 300,
                                       "Server: %s\r\n"
                                       "Data: %s\r\n"
                                       "Access-Control-Allow-Origin: %s\r\n"
                                       "Content-Length: %zd\r\n"
                                       "Content-Type: %s\r\n"
                                       "Cache-Control: no-cache\r\n"
                                       "Expires: 0\r\n",
                                       node_ctx_p->father->cfg->server_name,
                                       abcdk_time_format(node_ctx_p->father->timefmt, NULL, node_ctx_p->father->loc_ctx),
                                       (a_c_a_o && *a_c_a_o ? a_c_a_o : "*"),
                                       size,
                                       type);

    abcdk_http_service_response_body_buffer(stream,data,size);
    abcdk_http_service_response_body(stream, NULL);

    return 0;
}

int abcdk_http_service_response_fd(abcdk_object_t *stream, uint32_t status, int fd, const char *type, const char *a_c_a_o)
{
    abcdk_object_t *obj;
    int chk;

    assert(stream != NULL && status >= 100 && fd >= 0);

    obj = abcdk_mmap_fd(fd,0,0,0);
    if(!obj)
        return -1;

    chk = abcdk_http_service_response(stream,status,obj,type,a_c_a_o);
    if(chk == 0)
        return 0;

    /*删除应答失败的。*/
    abcdk_object_unref(&obj);
    return -1;
}

int abcdk_http_service_check_auth(abcdk_object_t *stream)
{
    abcdk_http_service_node_t *node_ctx_p;
    abcdk_http_service_stream_t *stream_ctx_p;
    abcdk_option_t *auth_opt = NULL;
    const char *auth_p;
    int is_proxy = 0;
    int chk;

    assert(stream != NULL);

    stream_ctx_p = (abcdk_http_service_stream_t *)stream->pptrs[ABCDK_MAP_VALUE];
    node_ctx_p = (abcdk_http_service_node_t *)abcdk_asynctcp_get_userdata(stream_ctx_p->io_node);

    if (!node_ctx_p->father->cfg->auth_load_cb)
        return 0;

    auth_p = abcdk_http_service_request_header_get(stream, "Authorization");
    if (!auth_p)
    {
        auth_p = abcdk_http_service_request_header_get(stream, "Proxy-Authorization");
        is_proxy = (auth_p!=NULL);
    }

    /*如果客户端没携带授权，则提示客户端提交授权。*/
    if (!auth_p)
        goto ERR;

    abcdk_http_parse_auth(&auth_opt,auth_p);
    abcdk_option_set(auth_opt,"http-method",stream_ctx_p->method->pstrs[0]);

    chk = abcdk_http_check_auth(auth_opt,node_ctx_p->father->cfg->auth_load_cb,node_ctx_p->father->cfg->opaque);
    abcdk_option_free(&auth_opt);

    if(chk == 0)
        return 0;

ERR:

    abcdk_http_service_response_header(stream, (is_proxy ? 407 : 401), 1000,
                                       "Server: %s\r\n"
                                       "Data: %s\r\n"
                                       "Access-Control-Allow-Origin: *\r\n"
                                       "%s-Authenticate: Digest realm=\"%s\", charset=utf-8, nonce=\"%llu\"\r\n"
                                       "Content-Length: 0\r\n",
                                       node_ctx_p->father->cfg->server_name,
                                       abcdk_time_format(node_ctx_p->father->timefmt, NULL, node_ctx_p->father->loc_ctx),
                                       (is_proxy ? "Proxy" : "WWW"),
                                       node_ctx_p->father->cfg->server_realm,
                                       (uint64_t)abcdk_rand(&node_ctx_p->father->rand_seed));

    abcdk_http_service_response_body(stream,NULL);

    return -1;
}