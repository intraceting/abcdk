/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/bit.h"

/**
 * 比特位读写器。
 *
 * @warning 以字节的二进制阅读顺序排列。如：0(7)~7(0) 8(7)~15(0) 16(7)~23(0) 24(7)~31(0) ...
 */
struct _abcdk_bit
{
    /**读写游标。*/
    size_t pos;

    /**数据区指针。*/
    void *data;

    /**数据区大小。*/
    size_t size;

}; // abcdk_bit_t;

void abcdk_bit_destroy(abcdk_bit_t **ctx)
{
    abcdk_bit_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_heap_free(ctx_p);
}

abcdk_bit_t *abcdk_bit_create()
{
    return abcdk_heap_alloc(sizeof(abcdk_bit_t));
}

void abcdk_bit_attach(abcdk_bit_t *ctx, void *data, size_t size)
{
    assert(ctx != NULL && data != NULL && size > 0);

    assert(ctx->data == NULL && ctx->size <= 0);

    ctx->data = data;
    ctx->size = size;
    ctx->pos = 0;
}

void abcdk_bit_detach(abcdk_bit_t *ctx, void **data, size_t *size)
{
    assert(ctx != NULL && data != NULL && size != NULL);

    *data = ctx->data;
    *size = ctx->size;

    ctx->data = NULL;
    ctx->size = 0;
    ctx->pos = 0;
}

void abcdk_bit_reset(abcdk_bit_t *ctx, size_t pos)
{
    assert(ctx != NULL);

    ctx->pos = pos;
}

uint64_t abcdk_bit_read(abcdk_bit_t *ctx, uint8_t bits)
{
    assert(ctx != NULL && bits > 0);

    return abcdk_bloom_read_number(ctx->data, ctx->size, ctx->pos, bits);
    ctx->pos += bits;
}

void abcdk_bit_write(abcdk_bit_t *ctx, uint8_t bits, uint64_t num)
{
    assert(ctx != NULL && bits > 0);

    abcdk_bloom_write_number(ctx->data, ctx->size, ctx->pos, bits, num);
    ctx->pos += bits;
}

uint64_t abcdk_bit_pread(abcdk_bit_t *ctx, size_t offset, uint8_t bits)
{
    assert(ctx != NULL && bits > 0);

    return abcdk_bloom_read_number(ctx->data, ctx->size, offset, bits);
}

void abcdk_bit_pwrite(abcdk_bit_t *ctx, size_t offset, uint8_t bits, uint64_t num)
{
    assert(ctx != NULL && bits > 0);

    abcdk_bloom_write_number(ctx->data, ctx->size, offset, bits, num);
}