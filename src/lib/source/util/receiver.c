/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/receiver.h"

/** 接收器对象。*/
struct _abcdk_receiver
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_RECEIVER_MAGIC 123456789

    /** 引用计数器。*/
    volatile int refcount;
    
    /** 协议。*/
    int protocol;

    /** 缓存最大长度。*/
    size_t buf_max;

    /** 临时文件名。*/
    char tmp_file[PATH_MAX];
    
    /** 外部缓存对象。*/
    abcdk_object_t *tmp_obj;

    /** 内存块指针。*/
    void *buf;

    /* 读写偏移量。*/
    size_t offset;

    /** 容量。*/
    size_t capacity;

    /** 长度。*/
    size_t size;

    /**
     * 头部长度。
     *
     * @note 长度为0时，表示头部还未接收完整。
     */
    size_t hdr_len;
    
    /** 实体长度。*/
    size_t body_len;

    /** 头部环境信息。*/
    abcdk_object_t *hdr_envs[100];
    
};// abcdk_receiver_t;


void abcdk_receiver_unref(abcdk_receiver_t **ctx)
{
    abcdk_receiver_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->magic == ABCDK_RECEIVER_MAGIC);

    if (abcdk_atomic_fetch_and_add(&ctx_p->refcount, -1) != 1)
        return;

    assert(ctx_p->refcount == 0);

    ctx_p->magic = 0xcccccccc;

    /*创建时，如果绑定外部内部对象，则内部对象没有创建内存。*/
    if(ctx_p->tmp_obj)
        abcdk_object_unref(&ctx_p->tmp_obj);
    else
        abcdk_heap_freep(&ctx_p->buf);

    if(access(ctx_p->tmp_file,F_OK)==0)
        remove(ctx_p->tmp_file);

    for (int i = 0; i < 100; i++)
        abcdk_object_unref(&ctx_p->hdr_envs[i]);
    

    abcdk_heap_free(ctx_p);
}

abcdk_receiver_t *abcdk_receiver_refer(abcdk_receiver_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_receiver_t *abcdk_receiver_alloc(int protocol, size_t max, const char *tempdir)
{
    abcdk_receiver_t *ctx = NULL;
    
    ctx = abcdk_heap_alloc(sizeof(abcdk_receiver_t));
    if (!ctx)
        return NULL;

    ctx->magic = ABCDK_RECEIVER_MAGIC;
    ctx->refcount = 1;

    ctx->protocol = protocol;
    ctx->buf_max = max;
    ctx->tmp_obj = NULL;

    ctx->offset = 0;
    ctx->buf = NULL;
    ctx->size = 0;
    ctx->capacity = 0;

    ctx->hdr_len = 0;
    ctx->body_len = 0;
    ctx->hdr_envs[0] = ctx->hdr_envs[1] = NULL;
    
    if (tempdir && *tempdir)
    {
        if (access(tempdir, W_OK) != 0)
            goto final_error;

        strncpy(ctx->tmp_file, tempdir, PATH_MAX - 64);
        abcdk_dirdir(ctx->tmp_file, "abcdk-receiver-XXXXXX");

        ctx->tmp_obj = abcdk_mmap_tempfile(ctx->tmp_file, 4096, 1, 1);
        if (!ctx->tmp_obj)
            goto final_error;
    }

    return ctx;

final_error:

    abcdk_receiver_unref(&ctx);

    return NULL;
}

int _abcdk_receiver_resize(abcdk_receiver_t *ctx, size_t size)
{
    void *new_buf = NULL;
    int chk;

    assert(ctx != NULL && size > 0);

    /*新的大小与旧的大小一样时，不需要调整。*/
    if (ctx->size == size)
        goto final;

    ctx->size = size;

    /*新的容量与旧的容量一样时，不需要调整。*/
    if (ctx->capacity == ABCDK_MAX(ctx->size + 1, 4096UL))
        goto final;

    ctx->capacity = ABCDK_MAX(ctx->size + 1, 4096UL);

    if (ctx->tmp_obj)
    {
        /*内存数据落盘。*/
        chk = abcdk_msync(ctx->tmp_obj,0);
        if (chk != 0)
            return -1;

        /*重新映射文件。*/
        chk = abcdk_mremap(ctx->tmp_obj, ctx->capacity, 1, 1);
        if (chk != 0)
            return -1;

        /*绑定新内存。*/
        ctx->buf = ctx->tmp_obj->pptrs[0];
    }
    else
    {
        /*重新申请内存。*/
        new_buf = abcdk_heap_realloc(ctx->buf, ctx->capacity);
        if (!new_buf)
            return -1;

        /*绑定新内存。*/
        ctx->buf = new_buf;
    }

final:

    /*修正编移量。*/
    if (ctx->offset > ctx->size)
        ctx->offset = ctx->size;

    /*多出的一个字节赋值为0。*/
    ABCDK_PTR2U8(ctx->buf, ctx->size) = 0;

    return 0;
}

int _abcdk_receiver_check_weight_stream(abcdk_receiver_t *ctx,size_t *diff)
{
    if(ctx->offset <= 0)
    {
        *diff = ctx->buf_max;
        return 0;
    }

    return 1;
}

int _abcdk_receiver_check_weight_http(abcdk_receiver_t *ctx,size_t *diff)
{
    const char *p = NULL, *p_next = NULL;

    /*不能超过最大长度。*/
    if (ctx->buf_max < ctx->offset)
        return -1;

    /*如果未确定头部长度，则先定位头部长度。*/
    if (ctx->hdr_len <= 0)
    {
        /*至少需要四个字符。*/
        if (ctx->offset >= 4)
        {
            /*查找头部结束标志。*/
            if (abcdk_strncmp(ABCDK_PTR2I8PTR(ctx->buf, ctx->offset - 4), "\r\n\r\n", 4, 0) == 0)
            {
                ctx->hdr_len = ctx->offset;

                p_next = (char *)ctx->buf;
                for (int i = 0; i < ABCDK_ARRAY_SIZE(ctx->hdr_envs); i++)
                {
                    ctx->hdr_envs[i] = abcdk_strtok3(&p_next, "\r\n", 0);
                    if (!ctx->hdr_envs[i])
                        break;

                    if (ctx->body_len <= 0)
                    {
                        p = abcdk_match_env(ctx->hdr_envs[i]->pstrs[0], "Content-Length",':');
                        ctx->body_len = (p ? strtol(p, NULL, 0) : 0);
                    }
                }

                if (ctx->buf_max < ctx->hdr_len + ctx->body_len)
                    return -1;
                else
                    return _abcdk_receiver_check_weight_http(ctx,diff);
            }
        }

        /*增量扩展内存。*/
        *diff = 1;
        return 0;
    }
    else
    {
        /*不能超过实体限制。*/
        if (ctx->offset < ctx->hdr_len + ctx->body_len)
        {
            /*增量扩展内存。*/
            *diff = ABCDK_MIN(ctx->buf_max, ctx->hdr_len + ctx->body_len - ctx->offset);
            return 0;
        }

        return 1;
    }
}

int _abcdk_receiver_check_weight_chunked(abcdk_receiver_t *ctx,size_t *diff)
{
    /*
     * Chunked包格式。
     *
     * size(HEX)\r\n
     * data\r\n
    */


    /*不能超过最大长度。*/
    if (ctx->buf_max < ctx->offset)
        return -1;

    /*如果未确定头部长度，则先定位头部长度。*/
    if (ctx->hdr_len <= 0)
    {
        /*至少需要两个字符。*/
        if (ctx->offset >= 2)
        {
            /*查找行尾标志。*/
            if (abcdk_strncmp(ABCDK_PTR2I8PTR(ctx->buf, ctx->offset - 2), "\r\n", 2, 0) == 0)
            {
                ctx->hdr_len = ctx->offset;
                ctx->body_len = strtoll(ABCDK_PTR2I8PTR(ctx->buf, 0), NULL, 16);

                if (ctx->buf_max < ctx->hdr_len + ctx->body_len)
                    return -1;
                else 
                    return _abcdk_receiver_check_weight_chunked(ctx,diff);
            }
        }

        /*增量扩展内存。*/
        *diff = 1;
        return 0;
    }
    else
    {
        /*不能超过实体限制。*/
        if (ctx->offset < ctx->hdr_len + ctx->body_len + 2)
        {
            /*增量扩展内存。*/
            *diff = ABCDK_MIN(ctx->buf_max, ctx->hdr_len + ctx->body_len + 2 - ctx->offset);
            return 0;
        }

        return 1;
    }
}

int _abcdk_receiver_check_weight_rtcp(abcdk_receiver_t *ctx, size_t *diff)
{
    size_t cur_pos;
    int mk,len;

    /*
     * RTCP包格式。
     *
     * |$     |Channel |Length(Data) |Data    |
     * |1 Byte|1 Byte  |2 Bytes      |N Bytes |
    */

    /*不能超过最大长度。*/
    if (ctx->buf_max < ctx->offset)
        return -1;
        
    if (ctx->offset < 4)
    {
        *diff = 4 - ctx->offset;
        return 0;
    }

    mk = ABCDK_PTR2I8(ctx->buf, 0);
    len = abcdk_endian_b_to_h16(ABCDK_PTR2U16(ctx->buf, 2));

    if (mk != '$')
        return -1;

    if (ctx->buf_max < len + 4)
        return -1;

    if (ctx->offset < len + 4)
    {
        *diff = ABCDK_MIN(ctx->buf_max, len + 4 - ctx->offset);
        return 0;
    }

    return 1;
}

int _abcdk_receiver_check_weight_smb(abcdk_receiver_t *ctx, size_t *diff)
{
    uint32_t len;
    

    /*
     * SMB包格式。
     *
     * |Length(Data)    |Data    |
     * |4 Byte          |N Bytes |
     * 
     * 
     * Length：数据长度。注：不包含自身。
    */

    /*不能超过最大长度。*/
    if (ctx->buf_max < ctx->offset)
        return -1;
        
    if (ctx->offset < 4)
    {
        *diff = 4 - ctx->offset;
        return 0;
    }

    len = abcdk_bloom_read_number((uint8_t*)ctx->buf,4,0,32);

    if (len <= 0 || len > 0x00FFFFFF)
        return -1;

    if (ctx->buf_max < len + 4)
        return -1;

    if (ctx->offset < len + 4)
    {
        *diff = ABCDK_MIN(ctx->buf_max, len + 4 - ctx->offset);
        return 0;
    }

    return 1;
}

int _abcdk_receiver_check_weight_smb_half(abcdk_receiver_t *ctx, size_t *diff)
{
    uint32_t len;
    
    /*
     * SMB-Half包格式。
     *
     * |Length(Data)    |Data    |
     * |2 Byte          |N Bytes |
     * 
     * 
     * Length：数据长度。注：不包含自身。
    */

    /*不能超过最大长度。*/
    if (ctx->buf_max < ctx->offset)
        return -1;
        
    if (ctx->offset < 2)
    {
        *diff = 2 - ctx->offset;
        return 0;
    }

    len = abcdk_bloom_read_number((uint8_t*)ctx->buf,2,0,16);

    if (len <= 0 || len > 0x0000FFFF)
        return -1;

    if (ctx->buf_max < len + 2)
        return -1;

    if (ctx->offset < len + 2)
    {
        *diff = ABCDK_MIN(ctx->buf_max, len + 2 - ctx->offset);
        return 0;
    }

    return 1;
}

int _abcdk_receiver_check_weight(abcdk_receiver_t *ctx, size_t *diff)
{
    int chk;

    if (ctx->protocol == ABCDK_RECEIVER_PROTO_STREAM)
        chk = _abcdk_receiver_check_weight_stream(ctx, diff);
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_HTTP)
        chk = _abcdk_receiver_check_weight_http(ctx, diff);
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_CHUNKED)
        chk = _abcdk_receiver_check_weight_chunked(ctx, diff);
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_RTCP)
        chk = _abcdk_receiver_check_weight_rtcp(ctx, diff);
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_NBT)
        chk = _abcdk_receiver_check_weight_smb(ctx, diff);
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_NBT_HALF)
        chk = _abcdk_receiver_check_weight_smb_half(ctx, diff);
    else
        chk = -1;

    return chk;
}

int abcdk_receiver_append(abcdk_receiver_t *ctx, const void *data,size_t size,size_t *remain)
{
    size_t rsize = 0;
    size_t rall = 0,diff = 0;
    int chk;

    assert(ctx != NULL && data != NULL && size > 0 && remain != NULL);
    
    /*默认无剩余数据。*/
    *remain = 0;

    for (;;)
    {
        /*检测接收的数据是否完整。*/
        diff = 0;
        chk = _abcdk_receiver_check_weight(ctx,&diff);
        if (chk != 0)
            break;

        /*检查可用空间。*/
        if (ctx->size - ctx->offset < diff)
        {
            if (_abcdk_receiver_resize(ctx, ctx->size + diff) != 0)
                chk = -1;
        }

        if (chk != 0)
            break;

        rsize = ABCDK_MIN(ctx->size - ctx->offset, size - rall);
        rsize = ABCDK_MIN(rsize,diff);

        if (rsize <= 0)
            break;

        memcpy(ABCDK_PTR2VPTR(ctx->buf, ctx->offset), ABCDK_PTR2VPTR(data, rall), rsize);
        ctx->offset += rsize;
        rall += rsize;
    }

    /*计算剩于数据长度。*/
    *remain = size - rall;

    return chk;
}

const void *abcdk_receiver_data(abcdk_receiver_t *ctx, off_t off)
{
    void *p = NULL;
    size_t l = 0;

    assert(ctx != NULL);

    p = ctx->buf;
    l = ctx->offset;

    ABCDK_ASSERT(off < l, TT("偏移量必须小于数据长度。"));

    p = ABCDK_PTR2VPTR(p, off);

    return p;
}

size_t abcdk_receiver_length(abcdk_receiver_t *ctx)
{
    size_t l = 0;

    assert(ctx != NULL);

    l = ctx->offset;

    return l;
}

size_t abcdk_receiver_header_length(abcdk_receiver_t *ctx)
{
    size_t l = 0;

    assert(ctx != NULL);

    if (ctx->protocol == ABCDK_RECEIVER_PROTO_HTTP)
        l = ctx->hdr_len;
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_HTTP)
        l = ctx->hdr_len;
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_RTCP)
        l = 4;
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_NBT)
        l = 4;
    else 
        l = 0;

    return l;
}

size_t abcdk_receiver_body_length(abcdk_receiver_t *ctx)
{
    size_t l = 0;

    assert(ctx != NULL);
    
    if (ctx->protocol == ABCDK_RECEIVER_PROTO_HTTP)
        l = ctx->body_len;
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_CHUNKED)
        l = ctx->body_len;
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_RTCP)
        l = ctx->offset - 4;
    else if (ctx->protocol == ABCDK_RECEIVER_PROTO_NBT)
        l = ctx->offset - 4;
    else 
        l = ctx->offset;

    return l;
}

const void *abcdk_receiver_body(abcdk_receiver_t *ctx, off_t off)
{
    size_t hlen = 0, dlen = 0;

    assert(ctx != NULL);

    hlen = abcdk_receiver_header_length(ctx);
    dlen = abcdk_receiver_body_length(ctx);

    ABCDK_ASSERT(off < dlen, TT("偏移量必须小于实体长度。"));

    return abcdk_receiver_data(ctx, hlen + off);
}

const char *abcdk_receiver_header_line(abcdk_receiver_t *ctx, int line)
{
    assert(ctx != NULL && line >= 0);

    if (ctx->protocol != ABCDK_RECEIVER_PROTO_HTTP)
        return NULL;

    if (line < ABCDK_ARRAY_SIZE(ctx->hdr_envs))
    {
        if (ctx->hdr_envs[line])
            return ctx->hdr_envs[line]->pstrs[0];
    }

    return NULL;
}

const char *abcdk_receiver_header_line_getenv(abcdk_receiver_t *ctx, const char *name, uint8_t delim)
{
    const char *p = NULL;
    const char *v = NULL;

    assert(ctx != NULL && name != 0);

    if (ctx->protocol != ABCDK_RECEIVER_PROTO_HTTP)
        return NULL;

    for (int i = 1; ; i++)
    {
        p = abcdk_receiver_header_line(ctx, i);
        if (!p)
            return NULL;

        if (i == 0)
            continue;

        v = abcdk_match_env(p, name, delim);
        if (v)
            return v;
    }

    return NULL;
}

