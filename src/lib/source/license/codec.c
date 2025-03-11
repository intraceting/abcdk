/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/license/codec.h"

/*
 * ------------------------------------------------------------------|
 * |Licence Data                                                     |
 * ------------------------------------------------------------------|
 * |Version |Category |Product |Begin   |Durations |Nodes   |Reserve |
 * |1 byte  |1 byte   |1 byte  |8 Bytes |2 Bytes   |2 Bytes |2 Bytes |
 * ------------------------------------------------------------------|
*/

#define ABCDK_LICENSE_VER_1_SRC_LEN (1 + 1 + 1 + 8 + 2 + 2 + 2)


int abcdk_license_codec_encode(abcdk_object_t **dst, const abcdk_license_info_t *src, abcdk_license_codec_encrypt_cb encrypt_cb, void *opaque)
{
    uint8_t tmp_src[ABCDK_LICENSE_VER_1_SRC_LEN] = {0};
    abcdk_bit_t src_bit = {0};
    uint32_t src_chksum = ~0;
    abcdk_object_t *tmp_dst = NULL;
    abcdk_object_t *tmp_dst2 = NULL;

    assert(dst != NULL && src != NULL && encrypt_cb != NULL);

    src_bit.data = tmp_src;
    src_bit.size = ABCDK_LICENSE_VER_1_SRC_LEN;

    abcdk_bit_write_number(&src_bit, 8, 1);
    abcdk_bit_write_number(&src_bit, 8, src->category);
    abcdk_bit_write_number(&src_bit, 8, src->product);
    abcdk_bit_write_number(&src_bit, 64, src->begin);
    abcdk_bit_write_number(&src_bit, 16, src->duration);
    abcdk_bit_write_number(&src_bit, 16, src->node);
    abcdk_bit_seek(&src_bit,16);

    tmp_dst = encrypt_cb(src_bit.data, src_bit.pos / 8, 1, opaque);
    if (!tmp_dst)
        return -1;

    abcdk_object_unref(dst);
    *dst = abcdk_basecode_encode2(tmp_dst->pptrs[0], tmp_dst->sizes[0], 64);
    abcdk_object_unref(&tmp_dst);

    return (*dst ? 0 : -1);
}

int abcdk_license_codec_decode(abcdk_license_info_t *dst, const char *src, abcdk_license_codec_encrypt_cb encrypt_cb, void *opaque)
{
    abcdk_object_t *tmp_src = NULL;
    abcdk_object_t *tmp_src2 = NULL;
    abcdk_bit_t src_bit = {0};
    uint8_t src_salt[256] = {0};
    uint32_t old_chksum = 0;
    uint32_t new_chksum = 0;
    uint8_t ver;

    assert(dst != NULL && src != NULL && encrypt_cb != NULL);

    tmp_src = abcdk_basecode_decode2(src, strlen(src), 64);
    if (!tmp_src)
        goto ERR;

    tmp_src2 = encrypt_cb(tmp_src->pptrs[0], tmp_src->sizes[0], 0, opaque);
    if (!tmp_src2)
        goto ERR;

    if (ABCDK_LICENSE_VER_1_SRC_LEN != tmp_src2->sizes[0])
        goto ERR;

    src_bit.data = tmp_src2->pptrs[0];
    src_bit.size = tmp_src2->sizes[0];

    ver = abcdk_bit_read2number(&src_bit, 8);
    if (ver != 1)
        goto ERR;

    dst->category = abcdk_bit_read2number(&src_bit, 8);
    dst->product = abcdk_bit_read2number(&src_bit, 8);
    dst->begin = abcdk_bit_read2number(&src_bit, 64);
    dst->duration = abcdk_bit_read2number(&src_bit, 16);
    dst->node = abcdk_bit_read2number(&src_bit, 16);

    abcdk_object_unref(&tmp_src);
    abcdk_object_unref(&tmp_src2);
    return 0;

ERR:

    abcdk_object_unref(&tmp_src);
    abcdk_object_unref(&tmp_src2);
    return -1;
}
