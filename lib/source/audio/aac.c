/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#include "abcdk/audio/aac.h"


void abcdk_aac_extradata_serialize(const abcdk_aac_adts_header_t *hdr, void *data, size_t size)
{

}


void abcdk_aac_extradata_deserialize(const void *data, size_t size, abcdk_aac_adts_header_t *hdr)
{
    assert(hdr != NULL && data != NULL && (size == 2 || size == 5));

    hdr->profile = abcdk_bloom_read_number(data,size,0,5);
    if(hdr->profile == 31)
    {
        hdr->profile = 32 + abcdk_bloom_read_number(data,size,5,6);
        hdr->sample_rate_index = abcdk_bloom_read_number(data,size,11,4);
        if(hdr->sample_rate_index == 15)
            hdr->channel_cfg = abcdk_bloom_read_number(data,size,15+24,4); //跳过24bits自定义的采样率。
        else
            hdr->channel_cfg = abcdk_bloom_read_number(data,size,15,4); 
    }
    else
    {
        hdr->sample_rate_index = abcdk_bloom_read_number(data,size,5,4);
        if(hdr->sample_rate_index == 15)
            hdr->channel_cfg = abcdk_bloom_read_number(data,size,9+24,4); //跳过24bits自定义的采样率。
        else
            hdr->channel_cfg = abcdk_bloom_read_number(data,size,9,4); 
    }

    /*填充其它头部字段。*/
    hdr->syncword = 0xfff;
    hdr->id = 0;
    hdr->protection_absent = 1;
    hdr->adts_buffer_fullness = 0x7ff;
}

void abcdk_aac_adts_header_serialize(const abcdk_aac_adts_header_t *hdr, void *data, size_t size)
{
    assert(hdr != NULL && data != NULL && size >= 7);

    abcdk_bloom_write_number(data, size, 0, 12, hdr->syncword);
    abcdk_bloom_write_number(data, size, 12, 1, hdr->id);
    abcdk_bloom_write_number(data, size, 13, 2, hdr->layer);
    abcdk_bloom_write_number(data, size, 15, 1, hdr->protection_absent);
    abcdk_bloom_write_number(data, size, 16, 2, hdr->profile);
    abcdk_bloom_write_number(data, size, 18, 4, hdr->sample_rate_index);
    abcdk_bloom_write_number(data, size, 22, 1, hdr->private_bit);
    abcdk_bloom_write_number(data, size, 23, 3, hdr->channel_cfg);
    abcdk_bloom_write_number(data, size, 26, 1, hdr->original_copy);
    abcdk_bloom_write_number(data, size, 27, 1, hdr->home);
    abcdk_bloom_write_number(data, size, 28, 1, hdr->copyright_identification_bit);
    abcdk_bloom_write_number(data, size, 29, 1, hdr->copyright_identification_start);
    abcdk_bloom_write_number(data, size, 30, 13, hdr->aac_frame_length);
    abcdk_bloom_write_number(data, size, 43, 11, hdr->adts_buffer_fullness);
    abcdk_bloom_write_number(data, size, 54, 2, hdr->raw_data_blocks);
}

void abcdk_aac_adts_header_deserialize(const void *data, size_t size, abcdk_aac_adts_header_t *hdr)
{
    assert(hdr != NULL && data != NULL && size >= 7);

    hdr->syncword = abcdk_bloom_read_number(data, size, 0, 12);
    hdr->id = abcdk_bloom_read_number(data, size, 12, 1);
    hdr->layer = abcdk_bloom_read_number(data, size, 13, 2);
    hdr->protection_absent = abcdk_bloom_read_number(data, size, 15, 1);
    hdr->profile = abcdk_bloom_read_number(data, size, 16, 2);
    hdr->sample_rate_index = abcdk_bloom_read_number(data, size, 18, 4);
    hdr->private_bit = abcdk_bloom_read_number(data, size, 22, 1);
    hdr->channel_cfg = abcdk_bloom_read_number(data, size, 23, 3);
    hdr->original_copy = abcdk_bloom_read_number(data, size, 26, 1);
    hdr->home = abcdk_bloom_read_number(data, size, 27, 1);
    hdr->copyright_identification_bit = abcdk_bloom_read_number(data, size, 28, 1);
    hdr->copyright_identification_start = abcdk_bloom_read_number(data, size, 29, 1);
    hdr->aac_frame_length = abcdk_bloom_read_number(data, size, 30, 13);
    hdr->adts_buffer_fullness = abcdk_bloom_read_number(data, size, 43, 11);
    hdr->raw_data_blocks = abcdk_bloom_read_number(data, size, 54, 2);
}

static int _abcdk_aac_sample_rates_dict[16] = {
        96000, 88200, 64000, 48000, 44100, 32000,
        24000, 22050, 16000, 12000, 11025, 8000, 7350};

int abcdk_aac_sample_rates(int idx)
{
    if (idx >= 15)
        return _abcdk_aac_sample_rates_dict[4];

    return _abcdk_aac_sample_rates_dict[idx];
}

int abcdk_aac_sample_rates2index(int rates)
{
    for (int i = 0; i < 15; i++)
    {
        if (_abcdk_aac_sample_rates_dict[i] == rates)
            return i;
    }

    return 15;
}

static uint8_t _abcdk_aac_channels_dict[8] = {
    0, 1, 2, 3, 4, 5, 6, 8};

int abcdk_aac_channels(int idx)
{
    if (idx >= 8)
        return _abcdk_aac_channels_dict[2];

    return _abcdk_aac_channels_dict[idx];
}

int abcdk_aac_channels2config(int channels)
{
    for (int i = 0; i < 8; i++)
    {
        if (_abcdk_aac_channels_dict[i] == channels)
            return i;
    }

    return 8;
}