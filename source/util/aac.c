/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/aac.h"

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