/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/package.h"

/**简单的封包工具。*/
struct _abcdk_package
{
    /*缓存。*/
    abcdk_object_t *buf;

    /*IO。*/
    abcdk_bit_t io;
}; // abcdk_package_t;

void abcdk_package_destroy(abcdk_package_t **ctx)
{
    abcdk_package_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->buf);
    abcdk_heap_free(ctx_p);
}

abcdk_package_t *abcdk_package_create(size_t max)
{
    abcdk_package_t *ctx;

    assert(max > 0 && max <= 0x7fffffff);

    ctx = (abcdk_package_t *)abcdk_heap_alloc(sizeof(abcdk_package_t));
    if (!ctx)
        return NULL;

    ctx->buf = abcdk_object_alloc2(max);
    if (!ctx->buf)
        goto ERR;

    ctx->io.size = ctx->buf->sizes[0];
    ctx->io.data = ctx->buf->pptrs[0];

    return ctx;

ERR:

    abcdk_package_destroy(&ctx);
    return NULL;
}

abcdk_package_t *abcdk_package_load(const uint8_t *data,size_t size)
{
    abcdk_package_t *ctx;
    int8_t data_f;
    ssize_t data_l;
    ssize_t chk;

    assert(data != NULL && size >0);

    ctx = (abcdk_package_t *)abcdk_heap_alloc(sizeof(abcdk_package_t));
    if (!ctx)
        return NULL;

    data_f = abcdk_bloom_read_number(data, 4, 0, 1);
    data_l = abcdk_bloom_read_number(data, 4, 1, 31);

    if (data_f)
    {
#ifdef LZ4_VERSION_NUMBER
        /*预分配足够的明文空间。*/
        ctx->buf = abcdk_object_alloc2(data_l);
        if (!ctx->buf)
            goto ERR;

        /*解压。*/
        chk = abcdk_lz4_dec(ctx->buf->pptrs[0], ctx->buf->sizes[0], data + 4, size - 4);
        if (chk != data_l)
            goto ERR;
#else
        abcdk_trace_printf(LOG_WARNING, "因为原文被压缩过，所以需要解压缩工具才能加载，但是当前环境并未包含解压缩工具。");
        goto ERR;
#endif // LZ4_VERSION_NUMBER
    }
    else
    {
        /*去掉4字节头部，复制。*/
        ctx->buf = abcdk_object_copyfrom(data + 4, size - 4);
        if (!ctx->buf)
            goto ERR;
    }

    ctx->io.size = ctx->buf->sizes[0];
    ctx->io.data = ctx->buf->pptrs[0];

    return ctx;
ERR:

    abcdk_package_destroy(&ctx);
    return NULL;
}

abcdk_object_t *abcdk_package_dump(abcdk_package_t *ctx, int compress)
{
    static volatile int warning_tip = 0;
    abcdk_object_t *obj;
    int8_t data_f;
    ssize_t data_l, data_l2;

    assert(ctx != NULL);

    /*
     * |Compress |Length |Data    |
     * |1 bit    |31 bit |N bytes |
     *
     * Compress：0 明文，1 压缩。
     * Length：数据长度（不包括自身）。
     * Data：变长数据。
     */

    /*空的。*/
    if (ctx->io.pos <= 0)
        return NULL;

    /*补齐到一个字节。*/
    while (ctx->io.pos % 8 != 0)
        ctx->io.pos += 1;

    /*默认不压缩。*/
    data_f = 0;
    /*计算原文长度。*/
    data_l = ctx->io.pos / 8;

#ifdef LZ4_VERSION_NUMBER
    if (compress)
    {
        /*压缩。*/
        data_f = 1;

        obj = abcdk_object_alloc2(4 + data_l);
        if (!obj)
            return NULL;

        /*压缩数据，并放置到4字节头部之后。*/
        data_l2 = abcdk_lz4_enc(obj->pptrs[0] + 4, obj->sizes[0] - 4, ctx->buf->pptrs[0], data_l);
        if (data_l2 <= 0)
        {
            if (abcdk_atomic_compare_and_swap(&warning_tip, 0, 1))
                abcdk_trace_printf(LOG_WARNING, "由于原文的压缩率低于0％，因此直接打包原文，而且后续再发生相同的情况不在再提示。");

            /*不压缩。*/
            data_f = 0;
            /*复制数据到4字节头部之后。*/
            memcpy(obj->pptrs[0] + 4, ctx->buf->pptrs[0], data_l);
        }
        else
        {
            /*修正长度。*/
            obj->sizes[0] = 4 + data_l2;
        }
    }
    else
#else
    if (abcdk_atomic_compare_and_swap(&warning_tip, 0, 1))
        abcdk_trace_printf(LOG_WARNING, "当前环境未包含解压缩工具，忽略压缩参数。");
#endif // LZ4_VERSION_NUMBER
    {
        obj = abcdk_object_alloc2(4 + data_l);
        if (!obj)
            return NULL;

        /*复制数据到4字节头部之后。*/
        memcpy(obj->pptrs[0] + 4, ctx->buf->pptrs[0], data_l);
    }

    /*写入头部信息。*/
    abcdk_bloom_write_number(obj->pptrs[0], 4, 0, 1, data_f);
    abcdk_bloom_write_number(obj->pptrs[0], 4, 1, 31, data_l);

    return obj;
}

int abcdk_package_eof(abcdk_package_t *ctx)
{
    assert(ctx != 0);

    return abcdk_bit_eof(&ctx->io);
}

size_t abcdk_package_seek(abcdk_package_t *ctx, ssize_t offset)
{
    assert(ctx != 0);

    return abcdk_bit_seek(&ctx->io, offset);
}

uint64_t abcdk_package_read2number(abcdk_package_t *ctx, uint8_t bits)
{
    assert(ctx != 0);

    return abcdk_bit_read2number(&ctx->io, bits);
}

void abcdk_package_read2buffer(abcdk_package_t *ctx, uint8_t *buf, size_t size)
{
    assert(ctx != 0);

    abcdk_bit_read2buffer(&ctx->io, buf, size);
}

void abcdk_package_write_number(abcdk_package_t *ctx, uint8_t bits, uint64_t num)
{
    assert(ctx != 0);

    abcdk_bit_write_number(&ctx->io, bits, num);
}

void abcdk_package_write_buffer(abcdk_package_t *ctx, const uint8_t *buf, size_t size)
{
    assert(ctx != 0);

    abcdk_bit_write_buffer(&ctx->io, buf, size);
}

void abcdk_package_write_string(abcdk_package_t *ctx, const char *buf, size_t size)
{
    size_t len, differ;

    assert(ctx != 0 && buf != NULL && size > 0);

    len = ABCDK_MIN((size_t)strlen(buf), size);
    differ = size - len;

    abcdk_package_write_buffer(ctx, (uint8_t *)buf, len);

    /*不足部分填零。*/
    for (size_t i = 0; i < differ; i++)
        abcdk_package_write_number(ctx, 8, 0);
}

uint64_t abcdk_package_pread2number(abcdk_package_t *ctx, size_t offset, uint8_t bits)
{
    assert(ctx != 0);

    return abcdk_bit_pread2number(&ctx->io, offset, bits);
}

void abcdk_package_pread2buffer(abcdk_package_t *ctx, size_t offset, uint8_t *buf, size_t size)
{
    assert(ctx != 0);

    abcdk_bit_pread2buffer(&ctx->io, offset, buf, size);
}

void abcdk_package_pwrite_number(abcdk_package_t *ctx, size_t offset, uint8_t bits, uint64_t num)
{
    assert(ctx != 0);

    abcdk_bit_pwrite(&ctx->io, offset, bits, num);
}

void abcdk_package_pwrite_buffer(abcdk_package_t *ctx, size_t offset, const uint8_t *buf, size_t size)
{
    assert(ctx != 0);

    abcdk_bit_pwrite_buffer(&ctx->io, offset, buf, size);
}

void abcdk_package_pwrite_string(abcdk_package_t *ctx, size_t offset, const char *buf, size_t size)
{
    size_t len, differ;

    assert(ctx != 0 && buf != NULL && size > 0);

    len = ABCDK_MIN((size_t)strlen(buf), size);
    differ = size - len;

    abcdk_package_pwrite_buffer(ctx, offset, (uint8_t *)buf, len);

    /*不足部分填零。*/
    for (size_t i = 0; i < differ; i++)
        abcdk_package_pwrite_number(ctx, offset + len, 8, 0);
}