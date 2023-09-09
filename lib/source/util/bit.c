/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/bit.h"

int abcdk_bit_eof(abcdk_bit_t *ctx)
{
    assert(ctx != NULL);

    return ((ctx->pos < ctx->size * 8) ? 0 : 1);
}

uint64_t abcdk_bit_seek(abcdk_bit_t *ctx, ssize_t offset)
{
    uint64_t old;

    assert(ctx != NULL);

    if (offset < 0)
        assert(ctx->pos >= labs(offset));
    if (offset > 0)
        assert(ctx->pos + offset <= ctx->size * 8);

    old = ctx->pos;
    ctx->pos += offset;

    return old;
}

uint64_t abcdk_bit_read(abcdk_bit_t *ctx, uint8_t bits)
{
    uint64_t chk;

    assert(ctx != NULL && bits > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);

    chk = abcdk_bloom_read_number(ctx->data, ctx->size, ctx->pos, bits);
    ctx->pos += bits;

    return chk;
}

void abcdk_bit_write(abcdk_bit_t *ctx, uint8_t bits, uint64_t num)
{
    assert(ctx != NULL && bits > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);

    abcdk_bloom_write_number(ctx->data, ctx->size, ctx->pos, bits, num);
    ctx->pos += bits;
}

uint64_t abcdk_bit_pread(abcdk_bit_t *ctx, size_t offset, uint8_t bits)
{
    assert(ctx != NULL && bits > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);

    return abcdk_bloom_read_number(ctx->data, ctx->size, offset, bits);
}

void abcdk_bit_pwrite(abcdk_bit_t *ctx, size_t offset, uint8_t bits, uint64_t num)
{
    assert(ctx != NULL && bits > 0);
    assert(ctx->data != NULL && ctx->size > 0 && ctx->pos < ctx->size * 8);

    abcdk_bloom_write_number(ctx->data, ctx->size, offset, bits, num);
}