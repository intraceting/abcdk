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

    /** 头部最大长度。*/
    size_t header_max;

    /** 实体最大长度。*/
    size_t body_max;

    /** 实体的临时文件。*/
    char body_tmpfile[PATH_MAX];
    
    /** 头部缓冲区。*/
    abcdk_message_t *hdr_buf;
    
    /** 实体缓冲区。*/
    abcdk_message_t *body_buf;

    /** 头部分析当前的游标位置。*/
    size_t hdr_parse_pos;

    /** 头部分析当前的行位置。*/
    size_t hdr_parse_line_pos;

    /**
     * 头部长度。
     *
     * @note 长度为0时，表示头部还未接收完整。
     */
    size_t hdr_len;
    
    /** 实体长度。*/
    size_t body_len;

    /** 头部环境变量的指针。*/
    const char *hdr_envs[100];

    /** 是否为RTSP包。*/
    int is_rtsp;

    /** 请求行解析标志。0 未解析，1 已解析。*/
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

    /*删除实体的临时文件。*/
    if(req_p->body_tmpfile[0])
        remove(req_p->body_tmpfile);

    abcdk_message_unref(&req_p->hdr_buf);
    abcdk_message_unref(&req_p->body_buf);
    abcdk_object_unref(&req_p->method);
    abcdk_object_unref(&req_p->location);
    abcdk_object_unref(&req_p->version);
    abcdk_object_unref(&req_p->path);
    abcdk_object_unref(&req_p->params);
    abcdk_heap_free2((void**)&req_p);

}

abcdk_http_request_t *abcdk_http_request_refer(abcdk_http_request_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_http_request_t *abcdk_http_request_alloc(size_t header_max,size_t body_max,const char *body_buffer_file)
{
    abcdk_http_request_t *req = NULL;

    assert(header_max > 0 && body_max > 0);
    assert(body_buffer_file == NULL || (body_buffer_file != NULL && strlen(body_buffer_file) <= PATH_MAX - 6));

    req = abcdk_heap_alloc(sizeof(abcdk_http_request_t));
    if (!req)
        goto final_error;

    req->refcount = 1;
    
    req->header_max = header_max;
    req->body_max = body_max;

    if(body_buffer_file && body_buffer_file[0])
    {
        strncpy(req->body_tmpfile, body_buffer_file, PATH_MAX - 6);
        abcdk_dirdir(req->body_tmpfile, "XXXXXX");
    }

    req->hdr_buf = NULL;
    req->body_buf = 0;
    req->hdr_parse_pos = 0;
    req->hdr_parse_line_pos = 0;
    req->hdr_len = 0;
    req->body_len = 0;
    memset(req->hdr_envs,0,sizeof(req->hdr_envs));
    req->is_rtsp = 0;
    req->env0_parse_ok = 0;
    req->method = NULL;
    req->location = NULL;
    req->version = NULL;
    req->path = NULL;
    req->params = NULL;

    return req;

final_error:

    abcdk_http_request_unref(&req);

    return NULL;
}

int _abcdk_http_request_natural_unpack_cb(void *opaque, abcdk_message_t *msg)
{
    abcdk_http_request_t *req_p = NULL;
    void *msg_ptr;
    size_t msg_len, msg_off, cur_pos;
    const char *p;
    size_t all_len;

    req_p = (abcdk_http_request_t *)opaque;

    /*处理接收到的数据。*/
    msg_ptr = abcdk_message_data(msg);
    msg_len = abcdk_message_size(msg);
    msg_off = abcdk_message_offset(msg);

    /*如果未确定头部长度，则先定位头部长度。*/
    if (req_p->hdr_len <= 0)
    {
        /*从上次结束位置开始找头部结束标志，目地是判断头部长度和实体长度。*/
        cur_pos = req_p->hdr_parse_pos;
        while (++cur_pos < msg_off)
        {
            /*找行尾标志。*/
            if (ABCDK_PTR2I8(msg_ptr, cur_pos) != '\n')
                continue;

            /*判断是否为头部结束标志(\r\n)。*/
            if (cur_pos - req_p->hdr_parse_pos == 1 &&
                ABCDK_PTR2I8(msg_ptr, req_p->hdr_parse_pos) == '\r' &&
                ABCDK_PTR2I8(msg_ptr, cur_pos) == '\n')
            {
                req_p->hdr_len = cur_pos + 1;//索引才是长度。
                req_p->hdr_parse_pos = 0;
                break;
            }
            else
            {
                /*记录当前行。*/
                req_p->hdr_envs[req_p->hdr_parse_line_pos] = ABCDK_PTR2U8PTR(msg_ptr, req_p->hdr_parse_pos);
                ABCDK_PTR2I8(msg_ptr, cur_pos - 1) = '\0';
                ABCDK_PTR2I8(msg_ptr, cur_pos) = '\0';

                /*尝试获取请求体长度。*/
                if (req_p->body_len <= 0)
                {
                    p = abcdk_http_match_env(req_p->hdr_envs[req_p->hdr_parse_line_pos], "Content-Length");
                    req_p->body_len = (p ? strtol(p, NULL, 0) : 0);
                }

                /*下一行。*/
                req_p->hdr_parse_line_pos += 1;
                req_p->hdr_parse_pos = ++cur_pos;
            }

            /*不支持超过100行的头部。*/
            if (req_p->hdr_parse_line_pos == 100)
                return -1;
        }

        /*请求头不完整的话，继续等待。*/
        if (req_p->hdr_len <= 0)
        {
            if (msg_off >= req_p->header_max)
                return -1;

            abcdk_message_expand(msg, 1);
            return 0;
        }

        return 1;
    }
    else
    {
        if (msg_off < req_p->body_len)
        {
            /*当实体不在外部缓存时，增量扩展内存。*/
            if (!req_p->body_tmpfile[0])
                abcdk_message_expand(msg, ABCDK_MIN(524288, req_p->body_len - msg_len));

            return 0;
        }

        return 1;
    }
    
}

int _abcdk_http_request_append_natural(abcdk_http_request_t *req, const void *data, size_t size, size_t *remain)
{
    int fd = -1;
    abcdk_object_t *body_tmp = NULL;
    int chk;

    /*如果未确定头部长度，则先定位头部长度。*/
    if (req->hdr_len <= 0)
    {
        /*准备头部的缓存。*/
        if (!req->hdr_buf)
        {
            req->hdr_buf = abcdk_message_alloc(1);
            if(!req->hdr_buf)
                return -1;

            abcdk_message_protocol_t prot = {req, _abcdk_http_request_natural_unpack_cb};
            abcdk_message_protocol_set(req->hdr_buf, &prot);
        }

        chk = abcdk_message_recv(req->hdr_buf,data, size, remain);
        if (chk != 1)
            return chk;

        /*可能无实体。*/
        if (req->body_len <= 0)
            return 1;
        else
            return 0;
    }
    else
    {
        if (req->body_max < req->body_len)
            return -1;
        
        /*准备实体的缓存。*/
        if(!req->body_buf)
        {
            /*
             * 1：如果缓存到文件，则一次性申请足够的空间。
             * 2：如果缓存到内存，则动态申请空间。
            */
            if (req->body_tmpfile[0])
                req->body_buf = abcdk_message_mmap_tempfile(req->body_tmpfile, req->body_len, 1);
            else
                req->body_buf = abcdk_message_alloc(ABCDK_MIN(524288,req->body_len));
            
            if(!req->body_buf)
                return -1;

            abcdk_message_protocol_t cb = {req, _abcdk_http_request_natural_unpack_cb};
            abcdk_message_protocol_set(req->body_buf, &cb);
        }

        chk = abcdk_message_recv(req->body_buf,data, size, remain);
        return chk;
    }
}

int _abcdk_http_request_rtsp_unpack_cb(void *opaque, abcdk_message_t *msg)
{
    abcdk_http_request_t *req_p = NULL;
    void *msg_ptr;
    size_t msg_len, msg_off, cur_pos;
    int len;

    req_p = (abcdk_http_request_t *)opaque;

    /*处理接收到的数据。*/
    msg_ptr = abcdk_message_data(msg);
    msg_len = abcdk_message_size(msg);
    msg_off = abcdk_message_offset(msg);

    if(msg_off < 4)
        return 0;

    len = abcdk_endian_b_to_h16(ABCDK_PTR2U16(msg_ptr, 2));

    if (req_p->body_max < len + 4)
        return -1;

    if (msg_off < len + 4)
    {
        abcdk_message_expand(msg, ABCDK_MIN(65536, len + 4 - msg_len));
        return 0;
    }

    return 1;
}

int _abcdk_http_request_append_rtsp(abcdk_http_request_t *req, const void *data, size_t size, size_t *remain)
{
    /*
     * |$     |Channel |Length(Data) |Data    |
     * |1 Byte|1 Byte  |2 Bytes      |N Bytes |
    */
    int chk;

    /*准备实体的缓存。*/
    if (!req->body_buf)
    {
        req->body_buf = abcdk_message_alloc(4);
        if (!req->body_buf)
            return -1;

        abcdk_message_protocol_t cb = {req, _abcdk_http_request_rtsp_unpack_cb};
        abcdk_message_protocol_set(req->body_buf, &cb);
    }

    chk = abcdk_message_recv(req->body_buf, data, size, remain);
    return chk;
}

int _abcdk_http_request_append_unknown(abcdk_http_request_t *req, const void *data, size_t size)
{
    int chk;

    if (req->body_max < size)
        return -1;

    if(req->body_buf)
        return -1;

    /*复制数据。*/
    req->body_buf = abcdk_message_copy(data, size);
    if (!req->body_buf)
        return -1;

    return 1;
}

int abcdk_http_request_append(abcdk_http_request_t *req, const void *data, size_t size, size_t *remain)
{
    int chk;

    assert(req != NULL && data != NULL && size > 0 && remain != NULL);

    if (remain)
    {
        /*如果出错，那么无剩余的数据。*/
        *remain = 0;

        /*如果还没有头部数据，并且第一个字符为$，按RTP协议解析流媒体数据包。*/
        if (req->hdr_len == 0 && ABCDK_PTR2I8(data, 0) == '$')
            req->is_rtsp = 1;

        /*RTSP包*/
        if (req->is_rtsp)
            chk = _abcdk_http_request_append_rtsp(req, data, size, remain);
        else
            chk = _abcdk_http_request_append_natural(req, data, size, remain);
    }
    else
    {
        chk = _abcdk_http_request_append_unknown(req, data, size);
    }

    return chk;
}

const void *abcdk_http_request_body(abcdk_http_request_t *req, off_t off)
{
    const void *p = NULL;
    size_t f = 0;

    assert(req != NULL);

    if (req->body_buf)
    {
        p = abcdk_message_data(req->body_buf);
        f = abcdk_message_offset(req->body_buf);
    }

    ABCDK_ASSERT(off <= f,"偏移量必须小于实体长度。");

    p = ABCDK_PTR2VPTR(p, off);

    return p;
}

size_t abcdk_http_request_body_length(abcdk_http_request_t *req)
{
    assert(req != NULL);

    if(req->body_buf)
        return abcdk_message_offset(req->body_buf);

    return 0;
}

const char *abcdk_http_request_env(abcdk_http_request_t *req, int line)
{
    assert(req != NULL && line >= 0);

    if (line < sizeof(req->hdr_envs))
        return (char *)req->hdr_envs[line];

    return NULL;
}

const char *abcdk_http_request_getenv(abcdk_http_request_t *req, const char *name)
{
    const char *p = NULL;
    const char *v = NULL;

    assert(req != NULL && name != 0);

    /*从1开始。*/
    for (int i = 1; i < 100; i++)
    {
        p = abcdk_http_request_env(req, i);
        if (!p)
            break;

        v = abcdk_http_match_env(p, name);
        if(v)
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

    if(!req->hdr_envs[0])
        return;

    p_next = req->hdr_envs[0];

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
    req->params = abcdk_strtok3(&p_next, "\r",0);
    if (!req->params)
        return;
    
    return;
}

const char *abcdk_http_request_method(abcdk_http_request_t *req)
{
    assert(req != NULL);

    _abcdk_http_request_parse_env0(req);

    if(req->method)
        return req->method->pstrs[0];

    return NULL;
}

const char *abcdk_http_request_location(abcdk_http_request_t *req)
{
    assert(req != NULL);

    _abcdk_http_request_parse_env0(req);

    if(req->location)
        return req->location->pstrs[0];

    return NULL;
}

const char *abcdk_http_request_version(abcdk_http_request_t *req)
{
    assert(req != NULL);

    _abcdk_http_request_parse_env0(req);

    if(req->version)
        return req->version->pstrs[0];

    return NULL;
}

const char *abcdk_http_request_path(abcdk_http_request_t *req)
{
    assert(req != NULL);

    _abcdk_http_request_parse_env0(req);

    if(req->path)
        return req->path->pstrs[0];

    return NULL;
}

const char *abcdk_http_request_params(abcdk_http_request_t *req)
{
    assert(req != NULL);

    _abcdk_http_request_parse_env0(req);

    if(req->params)
        return req->params->pstrs[0];

    return NULL;
}