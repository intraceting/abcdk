/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#include "abcdk/util/bit.h"

int abcdk_bit_eof(abcdk_bit_t *ctx)
{
    assert(ctx != NULL);

    return ((ctx->pos < ctx->size * 8) ? 0 : 1);
}

size_t abcdk_bit_seek(abcdk_bit_t *ctx, ssize_t offset)
{
    size_t old;

    assert(ctx != NULL);

    if (offset < 0)
        assert(ctx->pos >= labs(offset));
    if (offset > 0)
        assert(ctx->pos + offset <= ctx->size * 8);

    old = ctx->pos;
    ctx->pos += offset;

    return old;
}

uint64_t abcdk_bit_read2number(abcdk_bit_t *ctx, uint8_t bits)
{
    uint64_t chk;

    assert(ctx != NULL && bits > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);
    assert((ctx->pos + bits) <= (ctx->size * 8));

    chk = abcdk_bloom_read_number(ctx->data, ctx->size, ctx->pos, bits);
    ctx->pos += bits;

    return chk;
}

void abcdk_bit_read2buffer(abcdk_bit_t *ctx, uint8_t *buf, size_t size)
{
    assert(ctx != NULL && buf != NULL && size > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);
    assert(ctx->pos % 8 == 0 && (ctx->pos / 8 + size) <= ctx->size);

    memcpy(buf, (uint8_t *)ctx->data + (ctx->pos / 8), size);
    ctx->pos += (size * 8);
}

abcdk_object_t *abcdk_bit_read2object(abcdk_bit_t *ctx, size_t size)
{
    abcdk_object_t *buf;

    assert(ctx != NULL && size > 0);

    buf = abcdk_object_alloc2(size);
    if(!buf)
        return NULL;

    abcdk_bit_read2buffer(ctx,buf->pptrs[0],buf->sizes[0]);

    return buf;
}

void abcdk_bit_write_number(abcdk_bit_t *ctx, uint8_t bits, uint64_t num)
{
    assert(ctx != NULL && bits > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);
    assert((ctx->pos + bits) <= (ctx->size * 8));

    abcdk_bloom_write_number(ctx->data, ctx->size, ctx->pos, bits, num);
    ctx->pos += bits;
}

void abcdk_bit_write_buffer(abcdk_bit_t *ctx, const uint8_t *buf, size_t size)
{
    assert(ctx != NULL && buf != NULL && size > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);
    assert(ctx->pos % 8 == 0 && (ctx->pos / 8 + size) <= ctx->size);

    memcpy((uint8_t *)ctx->data + (ctx->pos / 8), buf, size);
    ctx->pos += (size * 8);
}

uint64_t abcdk_bit_pread2number(abcdk_bit_t *ctx, size_t offset, uint8_t bits)
{
    assert(ctx != NULL && bits > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);
    assert((ctx->pos + offset + bits) <= (ctx->size * 8));

    return abcdk_bloom_read_number(ctx->data, ctx->size, offset, bits);
}

void abcdk_bit_pread2buffer(abcdk_bit_t *ctx, size_t offset, uint8_t *buf,size_t size)
{
    assert(ctx != NULL && buf != NULL && size > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);
    assert(ctx->pos % 8 == 0 && ((ctx->pos+offset) / 8 + size) <= ctx->size);

    memcpy(buf, (uint8_t *)ctx->data + ((ctx->pos +offset) / 8), size);
}

void abcdk_bit_pwrite_number(abcdk_bit_t *ctx, size_t offset, uint8_t bits, uint64_t num)
{
    assert(ctx != NULL && bits > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);
    assert((ctx->pos + offset + bits) <= (ctx->size * 8));

    abcdk_bloom_write_number(ctx->data, ctx->size, offset, bits, num);
}

void abcdk_bit_pwrite_buffer(abcdk_bit_t *ctx, size_t offset, const uint8_t *buf, size_t size)
{
    assert(ctx != NULL && buf != NULL && size > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);
    assert(ctx->pos % 8 == 0 && ((ctx->pos + offset) * 8 + size) <= ctx->size);

    memcpy((uint8_t *)ctx->data + ((ctx->pos + offset) * 8), buf, size);
    ctx->pos += (size * 8);
}
