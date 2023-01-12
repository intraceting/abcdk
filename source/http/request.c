/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/http/request.h"

struct _abcdk_http_request 
{
    /** 引用计数器。*/
    volatile int refcount;

    /** 协议。*/
    int protocol;
        
    /** 缓存区。*/
    abcdk_receiver_t *buf;

    /** 缓存最大长度。*/
    size_t buf_max;

    /** 头部接收当前的游标位置。*/
    size_t hdr_recv_pos;

    /** 头部接收当前的行位置。*/
    size_t hdr_recv_line_pos;

    /**
     * 头部长度。
     *
     * @note 长度为0时，表示头部还未接收完整。
     */
    size_t hdr_len;
    
    /** 实体长度。*/
    size_t body_len;

    /** 头部解析标志。0 未解析，1 已解析。*/
    volatile int hdr_parse_ok;

    /** 头部环境信息。*/
    abcdk_object_t *envs[100];

    /** 零行解析标志。0 未解析，1 已解析。*/
    volatile int env0_parse_ok;

    /** 请求方法。*/
    abcdk_object_t *method;

    /** 定位符。*/
    abcdk_object_t *location;

    /** 协议和版本。*/
    abcdk_object_t *version;

    /** 路径。*/
    abcdk_object_t *path;

    /** 参数。*/
    abcdk_object_t *params;
    
};// abcdk_http_request_t;

void abcdk_http_request_unref(abcdk_http_request_t **req)
{
    abcdk_http_request_t *req_p = NULL;

    if (!req || !*req)
        return;

    req_p = *req;
    *req = NULL;

    if (abcdk_atomic_fetch_and_add(&req_p->refcount, -1) != 1)
        return;

    assert(req_p->refcount == 0);

    abcdk_receiver_unref(&req_p->buf);
    abcdk_object_unref(&req_p->method);
    abcdk_object_unref(&req_p->location);
    abcdk_object_unref(&req_p->version);
    abcdk_object_unref(&req_p->path);
    abcdk_object_unref(&req_p->params);

    abcdk_heap_free(req_p);
}

abcdk_http_request_t *abcdk_http_request_refer(abcdk_http_request_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_http_request_t *abcdk_http_request_alloc(int proto,size_t max,const char *tempdir)
{
    abcdk_http_request_t *req = NULL;

    assert(max > 0);

    req = abcdk_heap_alloc(sizeof(abcdk_http_request_t));
    if (!req)
        return NULL;

    req->refcount = 1;
    req->hdr_recv_pos = 0;
    req->hdr_recv_line_pos = 0;
    req->hdr_len = 0;
    req->body_len = 0;
    req->hdr_parse_ok = 0;
    req->envs[0] = req->envs[1] = NULL;
    req->env0_parse_ok = 0;
    req->method = NULL;
    req->location = NULL;
    req->version = NULL;
    req->path = NULL;
    req->params = NULL;

    req->protocol = proto;
    req->buf_max = max;
    req->buf = abcdk_receiver_alloc(tempdir);

    if (!req->buf)
        goto final_error;

    return req;

final_error:

    abcdk_http_request_unref(&req);

    return NULL;
}

int _abcdk_http_request_natural_unpack_cb(void *opaque, abcdk_receiver_t *msg,size_t *diff)
{
    abcdk_http_request_t *req_p = NULL;
    void *msg_ptr;
    size_t msg_len, msg_off, cur_pos;
    const char *p,*p2;
    size_t all_len;

    req_p = (abcdk_http_request_t *)opaque;

    /*处理接收到的数据。*/
    msg_ptr = abcdk_receiver_data(msg);
    msg_len = abcdk_receiver_size(msg);
    msg_off = abcdk_receiver_offset(msg);

    /*如果未确定头部长度，则先定位头部长度。*/
    if (req_p->hdr_len <= 0)
    {
        /*从上次结束位置开始找头部结束标志，目地是判断头部长度和实体长度。*/
        cur_pos = req_p->hdr_recv_pos;
        while (++cur_pos < msg_off)
        {
            /*查找行尾标志。*/
            if (ABCDK_PTR2I8(msg_ptr, cur_pos) != '\n')
                continue;

            /*判断是否为头部结束标志(\r\n)。*/
            if (cur_pos - req_p->hdr_recv_pos == 1 &&
                ABCDK_PTR2I8(msg_ptr, req_p->hdr_recv_pos) == '\r' &&
                ABCDK_PTR2I8(msg_ptr, cur_pos) == '\n')
            {
                req_p->hdr_len = cur_pos + 1;//索引才是长度。
                req_p->hdr_recv_pos = 0;
                break;
            }
            else
            {
                /*记录当前行。*/
                p2 = ABCDK_PTR2U8PTR(msg_ptr, req_p->hdr_recv_pos);

                /*尝试获取请求体长度。*/
                if (req_p->body_len <= 0)
                {
                    p = abcdk_http_match_env(p2, "Content-Length");
                    req_p->body_len = (p ? strtol(p, NULL, 0) : 0);
                }

                /*下一行。*/
                req_p->hdr_recv_line_pos += 1;
                req_p->hdr_recv_pos = ++cur_pos;
            }

            /*不支持超过100行的头部。*/
            if (req_p->hdr_recv_line_pos == 100)
                return -1;
        }

        /*检查请求头是否完整。。*/
        if (req_p->hdr_len > 0)
        {
            /*可能无实体。*/
            if (req_p->body_len <= 0)
                return 1;
            else
                return _abcdk_http_request_natural_unpack_cb(opaque,msg,diff);
        }
        else
        {
            /*不能超过最大长度。*/
            if (req_p->buf_max < msg_off)
                return -1;

            /*增量扩展内存。*/
            *diff = 1;
            return 0;
        }
    }
    else
    {
        /*不能超过实体限制。*/
        if (msg_off < req_p->hdr_len + req_p->body_len)
        {
            /*不能超过最大长度。*/
            if (req_p->buf_max < msg_off)
                return -1;

            /*增量扩展内存。*/
            *diff = ABCDK_MIN(524288, req_p->hdr_len + req_p->body_len - msg_off);
            return 0;
        }

        return 1;
    }
    
}

int _abcdk_http_request_append_natural(abcdk_http_request_t *req, const void *data, size_t size, size_t *remain)
{
    abcdk_receiver_protocol_set_simple(req->buf, req, _abcdk_http_request_natural_unpack_cb);

    return abcdk_receiver_recv(req->buf,data, size, remain);
}

int _abcdk_http_request_rtcp_unpack_cb(void *opaque, abcdk_receiver_t *msg,size_t *diff)
{
    abcdk_http_request_t *req_p = NULL;
    void *msg_ptr;
    size_t msg_len, msg_off, cur_pos;
    int mk,len;

    req_p = (abcdk_http_request_t *)opaque;

    /*处理接收到的数据。*/
    msg_ptr = abcdk_receiver_data(msg);
    msg_len = abcdk_receiver_size(msg);
    msg_off = abcdk_receiver_offset(msg);

    if (msg_off < 4)
    {
        *diff = 4 - msg_off;
        return 0;
    }

    mk = ABCDK_PTR2I8(msg_ptr, 0);
    len = abcdk_endian_b_to_h16(ABCDK_PTR2U16(msg_ptr, 2));

    if (mk != '$')
        return -1;

    if (req_p->buf_max < len + 4)
        return -1;

    if (msg_off < len + 4)
    {
        *diff = ABCDK_MIN(65536, len + 4 - msg_off);
        return 0;
    }

    return 1;
}

int _abcdk_http_request_append_rtcp(abcdk_http_request_t *req, const void *data, size_t size, size_t *remain)
{
    /*
     * RTCP包格式。
     *
     * |$     |Channel |Length(Data) |Data    |
     * |1 Byte|1 Byte  |2 Bytes      |N Bytes |
    */

    abcdk_receiver_protocol_set_simple(req->buf, req, _abcdk_http_request_rtcp_unpack_cb);

    return abcdk_receiver_recv(req->buf, data, size, remain);
}

int _abcdk_http_request_append_tunnel(abcdk_http_request_t *req, const void *data, size_t size, size_t *remain)
{
    return abcdk_receiver_recv(req->buf, data, size, remain);
}

int abcdk_http_request_append(abcdk_http_request_t *req, const void *data, size_t size, size_t *remain)
{
    int chk;

    assert(req != NULL && data != NULL && size > 0);

    /*默认无剩余的数据。*/
    *remain = 0;

    if (req->protocol == ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        chk = _abcdk_http_request_append_natural(req, data, size, remain);
    else if (req->protocol == ABCDK_HTTP_REQUEST_PROTO_RTCP)
        chk = _abcdk_http_request_append_rtcp(req, data, size, remain);
    else if (req->protocol == ABCDK_HTTP_REQUEST_PROTO_TUNNEL)
        chk = _abcdk_http_request_append_tunnel(req, data, size, remain);
    else
        chk = -1;

    return chk;
}

const void *abcdk_http_request_body(abcdk_http_request_t *req, off_t off)
{
    void *p = NULL;
    size_t l = 0;

    assert(req != NULL);

    if (req->protocol == ABCDK_HTTP_REQUEST_PROTO_NATURAL)
    {
        ABCDK_ASSERT(off < req->body_len, "偏移量必须小于实体长度。");

        p = abcdk_receiver_data(req->buf);
        p = ABCDK_PTR2VPTR(p, off + req->hdr_len);
    }
    else
    {
        l = abcdk_receiver_offset(req->buf);

        ABCDK_ASSERT(off < l, "偏移量必须小于实体长度。");

        p = abcdk_receiver_data(req->buf);
        p = ABCDK_PTR2VPTR(p, off);
    }

    return p;
}

size_t abcdk_http_request_body_length(abcdk_http_request_t *req)
{
    size_t l = 0;

    assert(req != NULL);
    
    if (req->protocol == ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        l = req->body_len;
    else 
        l = abcdk_receiver_offset(req->buf);

    return l;
}

void _abcdk_http_request_parse_hdr(abcdk_http_request_t *req)
{
    const char *p = NULL, *p_next = NULL;

    /*只解析一次。*/
    if (!abcdk_atomic_compare_and_swap(&req->hdr_parse_ok, 0, 1))
        return;

    p_next = (char *)abcdk_receiver_data(req->buf);

    for (int i = 0; i < ABCDK_ARRAY_SIZE(req->envs); i++)
    {
        req->envs[i] = abcdk_strtok3(&p_next, "\r\n",0);
        if (!req->envs[i])
            break;
    }
}

const char *abcdk_http_request_env(abcdk_http_request_t *req, int line)
{
    assert(req != NULL && line >= 0);

    if (req->protocol != ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        return NULL;

    _abcdk_http_request_parse_hdr(req);

    if (line < ABCDK_ARRAY_SIZE(req->envs))
    {
        if (req->envs[line])
            return req->envs[line]->pstrs[0];
    }

    return NULL;
}

const char *abcdk_http_request_getenv(abcdk_http_request_t *req, const char *name)
{
    const char *p = NULL;
    const char *v = NULL;

    assert(req != NULL && name != 0);

    if (req->protocol != ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        return NULL;

    for (int i = 1; ; i++)
    {
        p = abcdk_http_request_env(req, i);
        if (!p)
            return NULL;

        if (i == 0)
            continue;

        v = abcdk_http_match_env(p, name);
        if (v)
            return v;
    }

    return NULL;
}

void _abcdk_http_request_parse_env0(abcdk_http_request_t *req)
{
    const char *p = NULL, *p_next = NULL;
    size_t de_len;

    /*只解析一次。*/
    if(!abcdk_atomic_compare_and_swap(&req->env0_parse_ok,0,1))
        return;

    if(req->envs[0])
        p_next = req->envs[0]->pstrs[0];

    req->method = abcdk_strtok3(&p_next, " ",0);
    if(!req->method)
        return;

    req->location = abcdk_strtok3(&p_next, " ",0);
    if (!req->location)
        return;

    req->version = abcdk_strtok3(&p_next, " ",0);
    if (!req->version)
        return;

    p_next = req->location->pstrs[0];
    p = abcdk_strtok(&p_next, "?");
    if (!p)
        return;
    
    req->path = abcdk_object_alloc2(p_next - p);
    if (!req->path)
        return;

    abcdk_url_decode(p, p_next - p, req->path->pstrs[0], &req->path->sizes[0],0);

    /*去掉路径中的“..”和“.”，以防客户端构造特殊路径绕过WEB根目录。*/
    abcdk_abspath(req->path->pstrs[0]);
    /*修正路径长度。*/
    req->path->sizes[0] = strlen(req->path->pstrs[0]);

    /*可能无参数。*/
    if(!p_next || *p_next != '?')
        return;
        
    p_next += 1;
    req->params = abcdk_strtok3(&p_next, "\r\n",0);
    if (!req->params)
        return;
    
    return;
}

const char *abcdk_http_request_method(abcdk_http_request_t *req)
{
    assert(req != NULL);

    if (req->protocol != ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        return NULL;

    _abcdk_http_request_parse_env0(req);

    if (req->method)
        return req->method->pstrs[0];

    return NULL;
}

const char *abcdk_http_request_location(abcdk_http_request_t *req)
{
    assert(req != NULL);

    if (req->protocol != ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        return NULL;

    _abcdk_http_request_parse_env0(req);

    if (req->location)
        return req->location->pstrs[0];

    return NULL;
}

const char *abcdk_http_request_version(abcdk_http_request_t *req)
{
    assert(req != NULL);

    if (req->protocol != ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        return NULL;

    _abcdk_http_request_parse_env0(req);

    if (req->version)
        return req->version->pstrs[0];

    return NULL;
}

const char *abcdk_http_request_path(abcdk_http_request_t *req)
{
    assert(req != NULL);

    if (req->protocol != ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        return NULL;

    _abcdk_http_request_parse_env0(req);

    if (req->path)
        return req->path->pstrs[0];

    return NULL;
}

const char *abcdk_http_request_params(abcdk_http_request_t *req)
{
    assert(req != NULL);

    if (req->protocol != ABCDK_HTTP_REQUEST_PROTO_NATURAL)
        return NULL;

    _abcdk_http_request_parse_env0(req);

    if (req->params)
        return req->params->pstrs[0];

    return NULL;
}