/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/rtp/aac.h"

int abcdk_rtp_aac_revert(const void *data, size_t size, abcdk_comm_queue_t *q)
{
    abcdk_comm_message_t *msg;
    int r, len;

    /*
     * Header.
     * 
     *  0 1 2 3 4 5 6 7
     * +-+-+-+-+-+-+-+-+
     * |F|NRI|  Type   |
     * +---------------+
    */
    f = abcdk_bloom_read_number(data, size, 0, 1);
    nri = abcdk_bloom_read_number(data, size, 1, 2);
}