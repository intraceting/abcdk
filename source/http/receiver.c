/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/http/receiver.h"

struct _abcdk_http_receiver 
{
    /** 引用计数器。*/
    volatile int refcount;

    /** 协议。*/
    int protocol;
        
    /** 缓存区。*/
    abcdk_receiver_t *buf;

    /** 缓存最大长度。*/
    size_t buf_max;

    /**
     * 头部长度。
     *
     * @note 长度为0时，表示头部还未接收完整。
     */
    size_t hdr_len;
    
    /** 实体长度。*/
    size_t body_len;

    /** 头部环境信息。*/
    abcdk_object_t *envs[100];
    
};// abcdk_http_receiver_t;

void abcdk_http_receiver_unref(abcdk_http_receiver_t **rec)
{
    abcdk_http_receiver_t *rec_p = NULL;

    if (!rec || !*rec)
        return;

    rec_p = *rec;
    *rec = NULL;

    if (abcdk_atomic_fetch_and_add(&rec_p->refcount, -1) != 1)
        return;

    assert(rec_p->refcount == 0);

    abcdk_receiver_unref(&rec_p->buf);
    for (int i = 0; i < 100; i++)
        abcdk_object_unref(&rec_p->envs[i]);

    abcdk_heap_free(rec_p);
}

abcdk_http_receiver_t *abcdk_http_receiver_refer(abcdk_http_receiver_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_http_receiver_t *abcdk_http_receiver_alloc(int proto,size_t max,const char *tempdir)
{
    abcdk_http_receiver_t *rec = NULL;

    assert(max > 0);

    rec = abcdk_heap_alloc(sizeof(abcdk_http_receiver_t));
    if (!rec)
        return NULL;

    rec->refcount = 1;
    rec->hdr_len = 0;
    rec->body_len = 0;
    rec->envs[0] = rec->envs[1] = NULL;

    rec->protocol = proto;
    rec->buf_max = max;
    rec->buf = abcdk_receiver_alloc(tempdir);

    if (!rec->buf)
        goto final_error;

    return rec;

final_error:

    abcdk_http_receiver_unref(&rec);

    return NULL;
}

int _abcdk_http_receiver_natural_unpack_cb(void *opaque, const void *data, size_t size,size_t *diff)
{
    abcdk_http_receiver_t *rec_p = NULL;
    const char *p = NULL, *p_next = NULL;

    rec_p = (abcdk_http_receiver_t *)opaque;

    /*如果未确定头部长度，则先定位头部长度。*/
    if (rec_p->hdr_len <= 0)
    {
        /*至少需要四个字符。*/
        if (size >= 4)
        {
            /*查找头部结束标志。*/
            if (abcdk_strncmp(ABCDK_PTR2I8PTR(data, size - 4), "\r\n\r\n", 4, 0) == 0)
            {
                rec_p->hdr_len = size;

                p_next = (char *)data;
                for (int i = 0; i < ABCDK_ARRAY_SIZE(rec_p->envs); i++)
                {
                    rec_p->envs[i] = abcdk_strtok3(&p_next, "\r\n", 0);
                    if (!rec_p->envs[i])
                        break;

                    if (rec_p->body_len <= 0)
                    {
                        p = abcdk_http_match_env(rec_p->envs[i]->pstrs[0], "Content-Length");
                        rec_p->body_len = (p ? strtol(p, NULL, 0) : 0);
                    }
                }

                return _abcdk_http_receiver_natural_unpack_cb(opaque,data,size,diff);
            }
        }

        /*不能超过最大长度。*/
        if (rec_p->buf_max < size)
            return -1;

        /*增量扩展内存。*/
        *diff = 1;
        return 0;
    }
    else
    {
        /*不能超过实体限制。*/
        if (size < rec_p->hdr_len + rec_p->body_len)
        {
            /*不能超过最大长度。*/
            if (rec_p->buf_max < size)
                return -1;

            /*增量扩展内存。*/
            *diff = ABCDK_MIN(524288, rec_p->hdr_len + rec_p->body_len - size);
            return 0;
        }

        return 1;
    }
}

int _abcdk_http_receiver_append_natural(abcdk_http_receiver_t *rec, const void *data, size_t size, size_t *remain)
{
    abcdk_receiver_protocol_set_simple(rec->buf, rec, _abcdk_http_receiver_natural_unpack_cb);

    return abcdk_receiver_append(rec->buf,data, size, remain);
}

int _abcdk_http_receiver_chunked_unpack_cb(void *opaque, const void *data, size_t size,size_t *diff)
{
    abcdk_http_receiver_t *rec_p = NULL;

    rec_p = (abcdk_http_receiver_t *)opaque;

    /*如果未确定头部长度，则先定位头部长度。*/
    if (rec_p->hdr_len <= 0)
    {
        /*至少需要两个字符。*/
        if (size >= 2)
        {
            /*查找行尾标志。*/
            if (abcdk_strncmp(ABCDK_PTR2I8PTR(data, size - 2), "\r\n", 2, 0) == 0)
            {
                rec_p->hdr_len = size;
                rec_p->body_len = strtoll(ABCDK_PTR2I8PTR(data, 0), NULL, 16);

                return _abcdk_http_receiver_chunked_unpack_cb(opaque,data,size,diff);
            }
        }

        /*不能超过最大长度。*/
        if (rec_p->buf_max < size)
            return -1;

        /*增量扩展内存。*/
        *diff = 1;
        return 0;
    }
    else
    {
        /*不能超过实体限制。*/
        if (size < rec_p->hdr_len + rec_p->body_len + 2)
        {
            /*不能超过最大长度。*/
            if (rec_p->buf_max < size)
                return -1;

            /*增量扩展内存。*/
            *diff = ABCDK_MIN(524288, rec_p->hdr_len + rec_p->body_len + 2 - size);
            return 0;
        }

        return 1;
    }
}

int _abcdk_http_receiver_append_chunked(abcdk_http_receiver_t *rec, const void *data, size_t size, size_t *remain)
{
    /*
     * Chunked包格式。
     *
     * size(HEX)\r\n
     * data\r\n
    */

    abcdk_receiver_protocol_set_simple(rec->buf, rec, _abcdk_http_receiver_chunked_unpack_cb);

    return abcdk_receiver_append(rec->buf, data, size, remain);
}

int _abcdk_http_receiver_rtcp_unpack_cb(void *opaque, const void *data, size_t size,size_t *diff)
{
    abcdk_http_receiver_t *rec_p = NULL;
    size_t cur_pos;
    int mk,len;

    rec_p = (abcdk_http_receiver_t *)opaque;

    if (size < 4)
    {
        *diff = 4 - size;
        return 0;
    }

    mk = ABCDK_PTR2I8(data, 0);
    len = abcdk_endian_b_to_h16(ABCDK_PTR2U16(data, 2));

    if (mk != '$')
        return -1;

    if (rec_p->buf_max < len + 4)
        return -1;

    if (size < len + 4)
    {
        *diff = ABCDK_MIN(65536, len + 4 - size);
        return 0;
    }

    return 1;
}

int _abcdk_http_receiver_append_rtcp(abcdk_http_receiver_t *rec, const void *data, size_t size, size_t *remain)
{
    /*
     * RTCP包格式。
     *
     * |$     |Channel |Length(Data) |Data    |
     * |1 Byte|1 Byte  |2 Bytes      |N Bytes |
    */

    abcdk_receiver_protocol_set_simple(rec->buf, rec, _abcdk_http_receiver_rtcp_unpack_cb);

    return abcdk_receiver_append(rec->buf, data, size, remain);
}

int _abcdk_http_receiver_append_tunnel(abcdk_http_receiver_t *rec, const void *data, size_t size, size_t *remain)
{
    return abcdk_receiver_append(rec->buf, data, size, remain);
}

int abcdk_http_receiver_append(abcdk_http_receiver_t *rec, const void *data, size_t size, size_t *remain)
{
    int chk;

    assert(rec != NULL && data != NULL && size > 0);

    /*默认无剩余的数据。*/
    *remain = 0;

    if (rec->protocol == ABCDK_HTTP_RECEIVER_PROTO_NATURAL)
        chk = _abcdk_http_receiver_append_natural(rec, data, size, remain);
    else if (rec->protocol == ABCDK_HTTP_RECEIVER_PROTO_CHUNKED)
        chk = _abcdk_http_receiver_append_chunked(rec, data, size, remain);
    else if (rec->protocol == ABCDK_HTTP_RECEIVER_PROTO_RTCP)
        chk = _abcdk_http_receiver_append_rtcp(rec, data, size, remain);
    else if (rec->protocol == ABCDK_HTTP_RECEIVER_PROTO_TUNNEL)
        chk = _abcdk_http_receiver_append_tunnel(rec, data, size, remain);
    else
        chk = -1;

    return chk;
}

const void *abcdk_http_receiver_body(abcdk_http_receiver_t *rec, off_t off)
{
    void *p = NULL;
    size_t l = 0;

    assert(rec != NULL);

    if (rec->protocol == ABCDK_HTTP_RECEIVER_PROTO_NATURAL)
    {
        ABCDK_ASSERT(off < rec->body_len, "偏移量必须小于实体长度。");

        p = abcdk_receiver_data(rec->buf);
        p = ABCDK_PTR2VPTR(p, off + rec->hdr_len);
    }
    else if (rec->protocol == ABCDK_HTTP_RECEIVER_PROTO_CHUNKED)
    {
        ABCDK_ASSERT(off < rec->body_len, "偏移量必须小于实体长度。");

        p = abcdk_receiver_data(rec->buf);
        p = ABCDK_PTR2VPTR(p, off + rec->hdr_len);
    }
    else
    {
        l = abcdk_receiver_offset(rec->buf);

        ABCDK_ASSERT(off < l, "偏移量必须小于实体长度。");

        p = abcdk_receiver_data(rec->buf);
        p = ABCDK_PTR2VPTR(p, off);
    }

    return p;
}

size_t abcdk_http_receiver_body_length(abcdk_http_receiver_t *rec)
{
    size_t l = 0;

    assert(rec != NULL);
    
    if (rec->protocol == ABCDK_HTTP_RECEIVER_PROTO_NATURAL)
        l = rec->body_len;
    else if (rec->protocol == ABCDK_HTTP_RECEIVER_PROTO_CHUNKED)
        l = rec->body_len;
    else 
        l = abcdk_receiver_offset(rec->buf);

    return l;
}

const char *abcdk_http_receiver_header(abcdk_http_receiver_t *rec, int line)
{
    assert(rec != NULL && line >= 0);

    if (rec->protocol != ABCDK_HTTP_RECEIVER_PROTO_NATURAL)
        return NULL;

    if (line < ABCDK_ARRAY_SIZE(rec->envs))
    {
        if (rec->envs[line])
            return rec->envs[line]->pstrs[0];
    }

    return NULL;
}

const char *abcdk_http_receiver_getenv(abcdk_http_receiver_t *rec, const char *name)
{
    const char *p = NULL;
    const char *v = NULL;

    assert(rec != NULL && name != 0);

    if (rec->protocol != ABCDK_HTTP_RECEIVER_PROTO_NATURAL)
        return NULL;

    for (int i = 1; ; i++)
    {
        p = abcdk_http_receiver_header(rec, i);
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
