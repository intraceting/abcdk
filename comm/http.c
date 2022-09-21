/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "comm/http.h"

/** HTTP连接。*/
typedef struct _abcdk_comm_http
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_COMM_HTTP_MAGIC 20220920

    /** 
     * 标志。
     * 
     * 1：客户端。
     * 2：服务端。
     * 3：服务端(监听)。
    */
    int flag;

    /** 
     * 状态。
     * 
     * 0：断开或关闭。
     * 1：连接中(或监听中)。
     * 2：已连接。
    */
    volatile int status;

    /** 请求回调函数指针。*/
    abcdk_comm_http_request_cb request_cb;

    /** 
     * 链路状态。
     * 
     * 1：上行。
     * 2：下行。
    */
    volatile int link_state;

    /** 上行最大长度。*/
    size_t up_max_size;

    /** 接收缓冲区。*/
    abcdk_comm_message_t *in_buffer;

    /** 发送缓存。*/
    abcdk_comm_message_t *out_buffer;

    /** 发送队列。*/
    abcdk_comm_queue_t *out_queue;

    /** 请求分析当前游标位置。*/
    const char *req_parse_pos;

    /** 
     * 请求头部长度。
     * 
     * 长度为0时，表示头部还未接收完整。
    */
    size_t req_hdr_len;
    
    /** 请求方法。*/
    const char *req_method;

    /** 请求查询字符串。*/
    const char *req_query;

    /** 请求协议。*/
    const char *req_protocol;

    /** 请求环境变量。*/
    abcdk_object_t *req_env;

    /** 请求体长度。*/
    size_t req_body_len;

    /** 请求体。*/
    const void *req_body;

} abcdk_comm_http_t;

void _abcdk_comm_http_free(abcdk_comm_http_t **http)
{
    abcdk_comm_http_t *http_p = NULL;

    if(!http || !*http)
        return;

    http_p = *http;
    *http = NULL;

    abcdk_comm_message_unref(&http_p->in_buffer);
    abcdk_comm_message_unref(&http_p->out_buffer);
    abcdk_comm_queue_free(&http_p->out_queue);
    abcdk_object_unref(&http_p->req_env);
    abcdk_heap_free(http_p);
}

abcdk_comm_http_t *_abcdk_comm_http_alloc()
{
    abcdk_comm_http_t *http = NULL;

    http = (abcdk_comm_http_t *)abcdk_heap_alloc(sizeof(abcdk_comm_http_t));
    if (!http)
        return NULL;

    http->magic = ABCDK_COMM_HTTP_MAGIC;
    http->flag = 0;
    http->status = 1;
    http->link_state = 1;
    http->in_buffer = NULL;
    http->out_buffer = NULL;
    http->out_queue = abcdk_comm_queue_alloc();
    http->req_parse_pos = NULL;
    http->req_hdr_len = 0;
    http->req_env = NULL;
    http->req_body_len = 0;
    

    return http;
}

void _abcdk_comm_http_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_comm_http_t *http_p = NULL;

    if (!alloc->pptrs[0])
        return;

    http_p = (abcdk_comm_http_t *)alloc->pptrs[0];
    alloc->pptrs[0] = NULL;

    _abcdk_comm_http_free(&http_p);
}

abcdk_comm_node_t *abcdk_comm_http_alloc(abcdk_comm_t *ctx,size_t up_max_size)
{
    abcdk_comm_node_t *node = NULL;
    abcdk_comm_http_t *http = NULL;
    abcdk_object_t *append_p = NULL;

    assert(ctx != NULL && up_max_size >= 4096);
    
    node = abcdk_comm_alloc(ctx);
    if(!node)
        return NULL;

    http = _abcdk_comm_http_alloc();
    if(!http)
        goto final_error;

    /*绑定上行最大长度。*/
    http->up_max_size = up_max_size;
    
    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t*)http;
    abcdk_object_atfree(append_p,_abcdk_comm_http_destroy_cb,NULL);
    abcdk_object_unref(&append_p);

    return node;

final_error:

    abcdk_comm_unref(&node);

    return NULL;
}

const void *abcdk_comm_http_request_body(abcdk_comm_node_t *node)
{
    abcdk_comm_http_t *http_p = NULL;

    assert(node != NULL);

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_COMM_HTTP_MAGIC,"未通过http接口建立连接，不能调此接口。");

    return http_p->req_body;
}

const char *abcdk_comm_http_request_env(abcdk_comm_node_t *node, int line)
{
    abcdk_comm_http_t *http_p = NULL;

    assert(node != NULL && line >= 1);

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_COMM_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    if (line <= http_p->req_env->numbers)
        return (char *)http_p->req_env->pptrs[line - 1];

    return NULL;
}

const char *_abdk_comm_http_match_env(const char *line, const char *name)
{
    const char *s, *d;

    s = line;
    d = name;

    while (*s && *d)
    {
        if (toupper(*s) != toupper(*d))
            break;

        s += 1;
        d += 1;
    }

    if (*d == '\0' && *s == ':')
    {
        s += 1;
        while (*s)
        {
            if (!isspace(*s))
                break;

            s += 1;
        }

        return s;
    }

    return NULL;
}

const char *abcdk_comm_http_request_getenv(abcdk_comm_node_t *node, const char *name)
{
    abcdk_comm_http_t *http_p = NULL;
    const char *val = NULL;

    assert(node != NULL && name != NULL);

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_COMM_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    for (int i = 0; i < http_p->req_env->numbers; i++)
    {
        val = _abdk_comm_http_match_env((char*)http_p->req_env->pptrs[i],name);
        if(val)
            return val;
    }

    return NULL;
}

int abcdk_comm_http_response(abcdk_comm_node_t *node, const void *data, size_t len)
{
    abcdk_comm_http_t *http_p = NULL;
    abcdk_comm_message_t *msg = NULL;
    void *msg_ptr;
    size_t msg_len;
    int chk;

    assert(node != NULL && data != NULL && len > 0);

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_COMM_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    if(!abcdk_atomic_load(&http_p->status))
        return -2;

    msg = abcdk_comm_message_alloc(len);
    if (!msg)
        goto final_error;

    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    /*复制数据。*/
    memcpy(msg_ptr,data,len);

    chk = abcdk_comm_queue_push(http_p->out_queue, msg);
    if (chk != 0)
        goto final_error;

    if (abcdk_atomic_load(&http_p->status) == 2)
        abcdk_comm_send_watch(node);

    return 0;

final_error:

    abcdk_comm_message_unref(&msg);

    return -1;
}

int abcdk_comm_http_response2(abcdk_comm_node_t *node,abcdk_object_t *data)
{
    abcdk_comm_http_t *http_p = NULL;
    abcdk_comm_message_t *msg = NULL;
    int chk;

    assert(node != NULL && data != NULL);

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_COMM_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    if(!abcdk_atomic_load(&http_p->status))
        return -2;

    msg = abcdk_comm_message_alloc2(data);
    if (!msg)
        goto final_error;

    chk = abcdk_comm_queue_push(http_p->out_queue, msg);
    if (chk != 0)
        goto final_error;

    if (abcdk_atomic_load(&http_p->status) == 2)
        abcdk_comm_send_watch(node);

    return 0;

final_error:

    abcdk_comm_message_unref(&msg);

    return -1;
}

int abcdk_comm_http_response_end(abcdk_comm_node_t *node)
{
    abcdk_comm_http_t *http_p = NULL;

    assert(node != NULL);

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_COMM_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    if(!abcdk_atomic_load(&http_p->status))
        return -2;

    /*清理请求数据。*/
    abcdk_comm_message_unref(&http_p->in_buffer);
    http_p->req_parse_pos = NULL;
    http_p->req_hdr_len = 0;
    http_p->req_method = NULL;
    http_p->req_query = NULL;
    http_p->req_protocol = NULL;
    abcdk_object_unref(&http_p->req_env);
    http_p->req_body_len = 0;
    http_p->req_body = NULL;


    /*链路复用。*/
    abcdk_comm_recv_watch(node);

    return 0;
}

void _abcdk_comm_http_accept(abcdk_comm_node_t *node, abcdk_comm_node_t *listen)
{
    abcdk_comm_http_t *listen_http_p = NULL;
    abcdk_object_t *append_p = NULL;
    abcdk_comm_http_t *http_p = NULL;

    listen_http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(listen);

    http_p = _abcdk_comm_http_alloc();
    if (!http_p)
        return;

    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t*)http_p;
    abcdk_object_atfree(append_p,_abcdk_comm_http_destroy_cb,NULL);
    abcdk_object_unref(&append_p);

    /*标记为服务端。*/
    http_p->flag = 2;
    /*复制最大上行长度。*/
    http_p->up_max_size = listen_http_p->up_max_size;
    /*复制请求回调函数指针。*/
    http_p->request_cb = listen_http_p->request_cb;
    /*复制监听的用户环境指针。*/
    abcdk_comm_set_userdata(node,abcdk_comm_get_userdata(listen));
}

void _abcdk_comm_http_connect(abcdk_comm_node_t *node)
{
    abcdk_comm_http_t *http_p = NULL;

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);

    abcdk_atomic_store(&http_p->status, 2);

    /*已连接到远端，注册读写事件。*/
    abcdk_comm_recv_watch(node);
}

void _abcdk_comm_http_close(abcdk_comm_node_t *node)
{
    char sockname[NAME_MAX] = {0}, peername[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(node,sockname,peername);

    fprintf(stderr,"Disconnect: %s->%s\n",sockname,peername);
}

int _abcdk_comm_http_msg_protocol(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
{

}

void _abcdk_comm_http_input(abcdk_comm_node_t *node)
{
    abcdk_comm_http_t *http_p = NULL;
    void *msg_ptr;
    size_t msg_len;
    int chk;

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);

    /*准备接收数的缓存。*/
    if (!http_p->in_buffer)
    {
        http_p->in_buffer = abcdk_comm_message_alloc(4096);
        abcdk_comm_message_protocol_set(http_p->in_buffer, _abcdk_comm_http_msg_protocol);
    }

    /*没有可用的缓存时，通知超时，以关闭这个连接。*/
    if (!http_p->in_buffer)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }

    chk = abcdk_comm_message_recv(node, http_p->in_buffer);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_recv_watch(node);
        return;
    }

    /*处理接收到的数据。*/
    msg_ptr = abcdk_comm_message_data(http_p->in_buffer);
    msg_len = abcdk_comm_message_size(http_p->in_buffer);
}

void _abcdk_comm_http_output(abcdk_comm_node_t *node)
{
    abcdk_comm_http_t *http_p = NULL;
    int chk;

    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);

NEXT_MSG:

    /*如果发送缓存是空的，则从待发送队列取出一份。*/
    if (!http_p->out_buffer)
    {
        http_p->out_buffer = abcdk_comm_queue_pop(http_p->out_queue);
        if (!http_p->out_buffer)
            return;
    }

    chk = abcdk_comm_message_send(node, http_p->out_buffer);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_send_watch(node);
        return;
    }

    /*释放消息缓存，并继续发送。*/
    abcdk_comm_message_unref(&http_p->out_buffer);
    goto NEXT_MSG;
}

void _abcdk_comm_http_event_cb(abcdk_comm_node_t *node, uint32_t event, abcdk_comm_node_t *listen)
{
    if(event == ABCDK_COMM_EVENT_INPUT)
        _abcdk_comm_http_input(node);
    if(event == ABCDK_COMM_EVENT_OUTPUT)
        _abcdk_comm_http_output(node);
    else if(event == ABCDK_COMM_EVENT_CONNECT)
        _abcdk_comm_http_connect(node);
    else if(event == ABCDK_COMM_EVENT_ACCEPT)
        _abcdk_comm_http_accept(node,listen);
    else if(event == ABCDK_COMM_EVENT_CLOSE)
        _abcdk_comm_http_close(node);
}

int abcdk_comm_http_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_comm_http_request_cb request_cb)
{
    abcdk_comm_http_t *http_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && request_cb != NULL);
  
    http_p = (abcdk_comm_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_COMM_HTTP_MAGIC,"未通过http接口建立连接，不能调此接口。");
    
    /*初始化状态，标记为监听。*/
    http_p->flag = 3;
    http_p->status = 2;
    http_p->request_cb = request_cb;

    chk = abcdk_comm_listen(node, ssl_ctx, addr, _abcdk_comm_http_event_cb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}