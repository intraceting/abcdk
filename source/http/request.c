/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/http/request.h"

typedef struct _abcdk_http_request 
{
    /** 引用计数器。*/
    volatile int refcount;

    /** 请求的最大长度(头部+实体)。*/
    size_t max_size;

    /** 实体的临时文件。*/
    char body_tmpname[PATH_MAX];
    
    /** 头部缓冲区。*/
    abcdk_comm_message_t *hdr_buf;
    
    /** 实体缓冲区。*/
    abcdk_comm_message_t *body_buf;

    /** 头部分析当前的游标位置。*/
    size_t hdr_parse_pos;

    /** 头部分析当前的行位置。*/
    size_t hdr_parse_line_pos;

    /**
     * 头部长度。
     *
     * 长度为0时，表示头部还未接收完整。
     */
    size_t hdr_len;
    
    /** 实体长度。*/
    size_t body_len;

    /** 头部环境变量的指针。*/
    const char *hdr_envs[100];

}abcdk_http_request_t;

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
    if(req_p->body_tmpname[0])
        remove(req_p->body_tmpname);

    abcdk_comm_message_unref(&req_p->hdr_buf);
    abcdk_comm_message_unref(&req_p->body_buf);

}

abcdk_http_request_t *abcdk_http_request_refer(abcdk_http_request_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_http_request_t *abcdk_http_request_alloc(size_t max_size,const char *buffer_point)
{
    abcdk_http_request_t *req = NULL;

    assert(max_size > 0);
    assert(buffer_point == NULL || (buffer_point != NULL && strlen(buffer_point) <= PATH_MAX - 6));

    req = abcdk_heap_alloc(sizeof(abcdk_http_request_t));
    if (!req)
        goto final_error;

    req->refcount = 1;
    req->max_size = max_size;
    if(buffer_point && buffer_point[0])
    {
        strncpy(req->body_tmpname, buffer_point, PATH_MAX - 6);
        abcdk_dirdir(req->body_tmpname, "XXXXXX");
    }

    req->hdr_buf = NULL;
    req->body_buf = 0;
    req->hdr_parse_pos = 0;
    req->hdr_parse_line_pos = 0;
    req->hdr_len = 0;
    req->body_len = 0;
    memset(req->hdr_envs,0,sizeof(req->hdr_envs));

    return req;

final_error:

    abcdk_http_request_unref(&req);

    return NULL;
}

const void *abcdk_http_request_body(abcdk_http_request_t *req)
{
    assert(req != NULL);
    
    if(req->body_buf)
        return abcdk_comm_message_data(req->body_buf);

    return NULL;
}

const char *abcdk_http_request_env(abcdk_http_request_t *req, int line)
{
    assert(req != NULL && line >= 0);

    if (line < sizeof(req->hdr_envs))
        return (char *)req->hdr_envs[line];

    return NULL;
}

int _abcdk_http_request_unpack_cb(void *opaque, abcdk_comm_message_t *msg)
{
    abcdk_http_request_t *req_p = NULL;
    void *msg_ptr;
    size_t msg_len;
    size_t msg_off;
    size_t cur_pos;
    const char *p;
    size_t all_len;

    req_p = (abcdk_http_request_t *)opaque;

    /*处理接收到的数据。*/
    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);
    msg_off = abcdk_comm_message_offset(msg);

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
        }

        /*请求头不完整的话，继续等待。*/
        if (req_p->hdr_len <= 0)
        {
            if (msg_off >= req_p->max_size)
                return -1;

            abcdk_comm_message_expand(msg, 1);
            return 0;
        }

        return 1;
    }
    else
    {
        if (msg_off < req_p->body_len)
        {
            /*当实体不在外部缓存时，增量扩展内存。*/
            if (!req_p->body_tmpname[0])
                abcdk_comm_message_expand(msg, ABCDK_MIN(524288, req_p->body_len - msg_len));

            return 0;
        }

        return 1;
    }
    
}

int abcdk_http_request_append(abcdk_http_request_t *req, const void *data, size_t size, size_t *remain)
{
    int fd = -1;
    abcdk_object_t *body_tmp = NULL;
    int chk;

    assert(req != NULL && data != NULL && size > 0 && remain != NULL);

    /*如果出错，那么无剩余的数据。*/
    *remain = 0;

    /*如果未确定头部长度，则先定位头部长度。*/
    if (req->hdr_len <= 0)
    {
        /*准备头部的缓存。*/
        if (!req->hdr_buf)
        {
            req->hdr_buf = abcdk_comm_message_alloc(1);
            if(!req->hdr_buf)
                return -1;

            abcdk_comm_message_protocol_t prot = {req, _abcdk_http_request_unpack_cb};
            abcdk_comm_message_protocol_set(req->hdr_buf, &prot);
        }

        chk = abcdk_comm_message_recv2(req->hdr_buf,data, size, remain);
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
        if (req->max_size < req->hdr_len + req->body_len)
            return -1;
        
        /*准备实体的缓存。*/
        if(!req->body_buf)
        {
            if (req->body_tmpname[0])
            {
                fd = mkstemp(req->body_tmpname);
                if (fd >= 0)
                {
                    req->body_buf = abcdk_comm_message_alloc3(fd, req->body_len, 1);
                    abcdk_closep(&fd);
                }
            }
            else
            {
                req->body_buf = abcdk_comm_message_alloc(ABCDK_MIN(524288,req->body_len));
            }
            
            if(!req->body_buf)
                return -1;

            abcdk_comm_message_protocol_t prot = {req, _abcdk_http_request_unpack_cb};
            abcdk_comm_message_protocol_set(req->body_buf, &prot);
        }

        chk = abcdk_comm_message_recv2(req->body_buf,data, size, remain);
        return chk;
    }
}