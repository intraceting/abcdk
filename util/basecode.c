/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/basecode.h"

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
    if (c <= '9')
        return (uint8_t)(c - '2' + 26);
    else if (c <= 'Z')
        return (uint8_t)(c - 'A');
}

void abcdk_basecode_init(abcdk_basecode_t *ctx, uint8_t base)
{
    assert(ctx != NULL && base != 0);
    assert(base == 8 || base == 16 || base == 32 || base == 64 || base == 128 || base == 256);

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
    size_t bit_align = 1;
    size_t src_bits = 0, src_bits_align = 0;
    size_t base_bits = 0;
    size_t dlen = 0, dlen_align = 0, dst_bits_align = 0;
    uint8_t v = 0, a = 0;

    assert(ctx != NULL && src != NULL && slen > 0 && dst != NULL && dmaxlen > 0);
    assert(ctx->base == 8 || ctx->base == 16 || ctx->base == 32 || ctx->base == 64 || ctx->base == 128 || ctx->base == 256);
    assert(ctx->encode_table_cb != NULL);

    /*计算每个编码的bit数。*/
    for (size_t i = 1; i < ctx->base; i <<= 1)
        base_bits += 1;

    bit_align = abcdk_math_lcm(base_bits, 8);
    src_bits = slen * 8;
    src_bits_align = abcdk_align(src_bits, base_bits);
    dst_bits_align = abcdk_align(src_bits, bit_align);
    dlen_align = dst_bits_align / base_bits;

    /*不能超出缓存。*/
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

    dlen_align = dst_bits_align / base_bits;
    for (; dlen < dlen_align;)
        dst[dlen++] = ctx->pad;

final:

    return dlen;
}

ssize_t abcdk_basecode_decode(const abcdk_basecode_t *ctx,
                              const uint8_t *src, size_t slen,
                              uint8_t *dst, size_t dmaxlen)
{

}