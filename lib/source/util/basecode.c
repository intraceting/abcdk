/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/basecode.h"

uint8_t _abcdk_basecode_en_table64(uint8_t n)
{
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "abcdefghijklmnopqrstuvwxyz"
           "0123456789"
           "+/"[n];
}

uint8_t _abcdk_basecode_de_table64(uint8_t c)
{
    if (c == '+')
        return 62;
    else if (c == '/')
        return 63;
    else if (c <= '9')
        return (uint8_t)(c - '0' + 52);
    else if (c <= 'Z')
        return (uint8_t)(c - 'A');
    else if (c <= 'z')
        return (uint8_t)(c - 'a' + 26);
    
    return c;
}

uint8_t _abcdk_basecode_en_table32(uint8_t n)
{
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "234567"[n];
}

uint8_t _abcdk_basecode_de_table32(uint8_t c)
{
    if (c <= '7')
        return (uint8_t)(c - '2' + 26);
    else if (c <= 'Z')
        return (uint8_t)(c - 'A');

    return c;
}

void abcdk_basecode_init(abcdk_basecode_t *ctx, uint8_t base)
{
    assert(ctx != NULL && base != 0);

    ctx->base = base;

    if(ctx->base == 64)
    {
        ctx->encode_table_cb = _abcdk_basecode_en_table64;
        ctx->decode_table_cb = _abcdk_basecode_de_table64;
        ctx->pad = '=';
    }
    
    if(ctx->base == 32)
    {
        ctx->encode_table_cb = _abcdk_basecode_en_table32;
        ctx->decode_table_cb = _abcdk_basecode_de_table32;
        ctx->pad = '=';
    }
}

ssize_t abcdk_basecode_encode(const abcdk_basecode_t *ctx,
                              const uint8_t *src, size_t slen,
                              uint8_t *dst, size_t dmaxlen)
{
    size_t base_bits = 0;
    size_t bit_align = 1;
    size_t src_bits = 0, src_bits_align = 0;
    size_t dlen = 0, dlen_align = 0, dst_bits_align = 0;
    uint8_t v = 0, a = 0;

    assert(ctx != NULL && src != NULL && slen > 0 && dst != NULL && dmaxlen > 0);
    assert(ctx->base != 0 && ctx->encode_table_cb != NULL);

    /*计算每个编码的bit数。*/
    for (size_t i = 1; i < ctx->base; i <<= 1)
        base_bits += 1;

    /*最小公倍数。*/
    bit_align = abcdk_math_lcm(base_bits, 8);

    src_bits = slen * 8;
    src_bits_align = abcdk_align(src_bits, base_bits);
    dst_bits_align = abcdk_align(src_bits, bit_align);
    dlen_align = dst_bits_align / base_bits;

    /*不能超过目标缓存大小。*/
    assert(dlen_align <= dmaxlen);

    for (size_t i = 0; i < src_bits_align;)
    {
        v = a = 0;
        for (size_t j = 0; j < base_bits; j++, i++)
        {
            a = (i < src_bits) ? abcdk_bloom_read((uint8_t *)src, slen, i) : 0;
            v |= (a << (base_bits - j - 1));
        }

        dst[dlen++] = ctx->encode_table_cb(v);
    }

    /*补齐数据。*/
    for (; dlen < dlen_align;)
        dst[dlen++] = ctx->pad;

final:

    return dlen;
}

ssize_t abcdk_basecode_decode(const abcdk_basecode_t *ctx,
                              const uint8_t *src, size_t slen,
                              uint8_t *dst, size_t dmaxlen)
{
    size_t base_bits = 0;
    size_t src_bits = 0, src_bits_align = 0;
    size_t dlen = 0, dlen_align = 0,dst_bits_pos = 0;
    uint8_t v = 0, a = 0;

    assert(ctx != NULL && src != NULL && slen > 0 && dst != NULL && dmaxlen > 0);
    assert(ctx->base != 0 && ctx->decode_table_cb != NULL);

    /*计算每个编码的bit数。*/
    for (size_t i = 1; i < ctx->base; i <<= 1)
        base_bits += 1;

    src_bits = slen * base_bits;
    dlen_align = src_bits / 8;

    /*不能超过目标缓存大小。*/
    assert(dlen_align <= dmaxlen);

    for (size_t i = 0; i < slen; i++)
    {
        /*遇到补齐数据则停止。*/
        if (src[i] == ctx->pad)
            break;

        a = 0;
        v = ctx->decode_table_cb(src[i]);
        for (size_t j = 0; j < base_bits; j++)
        {
            a = v & (1 << (base_bits - j - 1));
            abcdk_bloom_write(dst, dmaxlen, dst_bits_pos++, a);
        }

        dlen = dst_bits_pos / 8;
    }

    return dlen;
}

abcdk_object_t *abcdk_basecode_encode2(const void *src, size_t len, uint8_t base)
{
    abcdk_object_t *dst = NULL;
    abcdk_basecode_t ctx;

    assert(src != NULL && len > 0);

    dst = abcdk_object_alloc2(len * 2);
    if (!dst)
        return NULL;

    abcdk_basecode_init(&ctx, base);
    dst->sizes[0] = abcdk_basecode_encode(&ctx, src, len, dst->pptrs[0], dst->sizes[0]);

    return dst;
}

abcdk_object_t *abcdk_basecode_decode2(const char *src,size_t len, uint8_t base)
{
    abcdk_object_t *dst = NULL;
    abcdk_basecode_t ctx;

    assert(src != NULL && len > 0);

    dst = abcdk_object_alloc2(len);
    if (!dst)
        return NULL;

    abcdk_basecode_init(&ctx, base);
    dst->sizes[0] = abcdk_basecode_decode(&ctx, (uint8_t*)src, len, dst->pptrs[0], dst->sizes[0]);

    return dst;
}