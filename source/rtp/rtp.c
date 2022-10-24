/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/rtp/rtp.h"

void abcdk_rtp_header_serialize(const abcdk_rtp_header_t *hdr, void *data, size_t size)
{
    assert(hdr != NULL && data != NULL && size > 0);
    assert(hdr->csrc_len * 4 + 12 <= size);

    abcdk_bloom_write_number(data,size,0,2,hdr->version);
    abcdk_bloom_write_number(data,size,2,1,hdr->padding);
    abcdk_bloom_write_number(data,size,3,1,hdr->extension);
    abcdk_bloom_write_number(data,size,4,4,hdr->csrc_len);
    abcdk_bloom_write_number(data,size,8,1,hdr->marker);
    abcdk_bloom_write_number(data,size,9,7,hdr->payload);
    abcdk_bloom_write_number(data,size,16,16,hdr->seq_no);
    abcdk_bloom_write_number(data,size,32,32,hdr->timestamp);
    abcdk_bloom_write_number(data,size,64,32,hdr->ssrc);

    for (int i = 0; i < hdr->csrc_len; i++)
        abcdk_bloom_write_number(data,size,96 + (i * 4 * 8),32,hdr->csrc[i]);
}

void abcdk_rtp_header_deserialize(const void *data, size_t size, abcdk_rtp_header_t *hdr)
{
    assert(hdr != NULL && data != NULL && size > 0);

    hdr->version = abcdk_bloom_read_number(data, size, 0, 2);
    hdr->padding = abcdk_bloom_read_number(data, size, 2, 1);
    hdr->extension = abcdk_bloom_read_number(data, size, 3, 1);
    hdr->csrc_len = abcdk_bloom_read_number(data, size, 4, 4);
    hdr->marker = abcdk_bloom_read_number(data, size, 8, 1);
    hdr->payload = abcdk_bloom_read_number(data, size, 9, 7);
    hdr->seq_no = abcdk_bloom_read_number(data, size, 16, 16);
    hdr->timestamp = abcdk_bloom_read_number(data, size, 32, 32);
    hdr->ssrc = abcdk_bloom_read_number(data, size, 64, 32);

    for (int i = 0; i < hdr->csrc_len; i++)
        hdr->csrc[i] = abcdk_bloom_read_number(data, size, 96 + (i * 32), 32);
}