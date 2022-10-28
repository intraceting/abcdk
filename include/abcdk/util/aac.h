/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_UTIL_AAC_H
#define ABCDK_UTIL_AAC_H

#include "abcdk/util/general.h"
#include "abcdk/util/bloom.h"

__BEGIN_DECLS

/** ADTS头部。*/
typedef struct _abcdk_aac_adts_header
{
    /** 同步字(ADTS帧识别码) 12bits。'1111 1111 1111'。*/
    uint16_t syncword;

    /** MPEG 标示符 1bit。0 for MPEG-4，1 for MPEG-2。*/
    uint8_t id;

    /** 2bits。'00'。*/
    uint8_t layer;

    /** CRC较验 1bit。1：无，0：有。 */
    uint8_t protection_absent;

    /** 级别 2bit。*/
    uint8_t profile;

    /** 采样频率索引 4bits。*/
    uint8_t sample_rate_index;

    /** 1bits。'0'。*/
    uint8_t private_bit;

    /** 声道数 3bit。*/
    uint8_t channel;

    /** 1bits。'0'。*/
    uint8_t original_copy;

    /** 1bits。'0'。*/
    uint8_t home;

    /** 1bits。'0'。*/
    uint8_t copyright_identification_bit;

    /** 1bits。'0'。*/
    uint8_t copyright_identification_start;

    /** 帧的长度 13bit。包括ADTS头和AAC原始流。*/
    uint16_t aac_frame_length;

    /** ？？？ 11bit。0x7FF：可变的码流。*/
    uint16_t adts_buffer_fullness;

    /** AAC数据块数量 2bit。'00'：一个数据块。*/
    uint8_t raw_data_blocks;
} abcdk_aac_adts_header_t;

/**序列化。*/
void abcdk_aac_adts_header_serialize(const abcdk_aac_adts_header_t *hdr, void *data, size_t size);

/**反序列化。*/
void abcdk_aac_adts_header_deserialize(const void *data, size_t size, abcdk_aac_adts_header_t *hdr);

__END_DECLS

#endif // ABCDK_UTIL_AAC_H