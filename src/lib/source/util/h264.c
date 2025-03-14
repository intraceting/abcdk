/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/h264.h"

void abcdk_h264_extradata_clean(abcdk_h264_extradata_t *extdata)
{
    assert(extdata != NULL);

    abcdk_object_unref(&extdata->sps);
    abcdk_object_unref(&extdata->pps);
    memset(extdata,0,sizeof(*extdata));
}

ssize_t abcdk_h264_extradata_serialize(const abcdk_h264_extradata_t *extdata, void *data, size_t size)
{
    return -1;
}

void abcdk_h264_extradata_deserialize(const void *data, size_t size, abcdk_h264_extradata_t *extdata)
{
    uint8_t d[3] = {0};
    const void *p1 = NULL, *p2 = NULL, *p3 = NULL;
    uint8_t type;
    abcdk_bit_t rbuf = {0};
    

    assert(data != NULL && size > 0 && extdata != NULL);

    rbuf.data = (void*)data;
    rbuf.size = size;
    rbuf.pos = 0;

    d[0] = abcdk_bit_pread(&rbuf, 0, 8);
    d[1] = abcdk_bit_pread(&rbuf, 8, 8);
    d[2] = abcdk_bit_pread(&rbuf, 16, 16);

    /*start code: 001 or 0001*/
    if(size > 3 && (!d[0] && !d[1] && d[2] == 1))
    {
        p1 = rbuf.data;
        p2 = NULL;
        p3 = rbuf.data + rbuf.size - 1;/*末尾指针要减1。*/

        for (int j = 0; j < 2; j++)
        {
            if (p1 > p3)
                return;

            p2 = abcdk_h2645_packet_split((void **)&p1, p3);
            if (p2 == NULL)
                return;

            type = ABCDK_PTR2U8(p2,0) & 0x1F;
            if(type == 7)
            {
                extdata->sps_num = 1;
                extdata->sps = abcdk_object_copyfrom(p2,p1 - p2);
            }
            else if(type == 8)
            {
                extdata->pps_num = 1;
                extdata->pps = abcdk_object_copyfrom(p2,p1 - p2);
            }
        }
    }
    else if(size > 7 && d[0] == 1)
    {
        extdata->version = abcdk_bit_read(&rbuf, 8);
        extdata->avc_profile = abcdk_bit_read(&rbuf, 8);
        extdata->avc_compatibility = abcdk_bit_read(&rbuf, 8);
        extdata->avc_level = abcdk_bit_read(&rbuf, 8);
        extdata->reserve1 = abcdk_bit_read(&rbuf, 6);
        extdata->nal_length_size = abcdk_bit_read(&rbuf, 2);
        extdata->reserve2 = abcdk_bit_read(&rbuf, 3);

        extdata->sps_num = abcdk_bit_read(&rbuf, 5);
        extdata->sps = abcdk_object_alloc3(rbuf.size, extdata->sps_num); // 足够的长度，保证不会溢出。
        for (int i = 0; i < extdata->sps_num; i++)
        {
            extdata->sps->sizes[i] = abcdk_bit_read(&rbuf, 16);
#if 0
            memcpy(extdata->sps->pptrs[i], ABCDK_PTR2VPTR(rbuf.data, rbuf.pos / 8), extdata->sps->sizes[i]);
            rbuf.pos += (extdata->sps->sizes[i] * 8);
#else
            abcdk_bit_read2buffer(&rbuf,extdata->sps->pptrs[i], extdata->sps->sizes[i]);
#endif 
        }

        extdata->pps_num = abcdk_bit_read(&rbuf, 8);
        extdata->pps = abcdk_object_alloc3(rbuf.size, extdata->pps_num); // 足够的长度，保证不会溢出。
        for (int i = 0; i < extdata->pps_num; i++)
        {
            extdata->pps->sizes[i] = abcdk_bit_read(&rbuf, 16);
#if 0
            memcpy(extdata->pps->pptrs[i], ABCDK_PTR2VPTR(rbuf.data, rbuf.pos / 8), extdata->pps->sizes[i]);
            rbuf.pos += (extdata->pps->sizes[i] * 8);
#else
            abcdk_bit_read2buffer(&rbuf,extdata->pps->pptrs[i], extdata->pps->sizes[i]);
#endif 
        }
    }
}

int abcdk_h264_idr(const void *data, size_t size)
{
    void *p_next = NULL, *p_end = NULL, *p = NULL;
    int nal_unit_type;
    int chk = 0;

    assert(data != NULL && size > 0);

    p_next = (void*)data;
    p_end = ABCDK_PTR2VPTR(data, size - 1);

    while (1)
    {
        if (p_next > p_end)
            break;

        p = (void *)abcdk_h2645_packet_split(&p_next, p_end);
        if (!p)
            break;

        nal_unit_type = abcdk_bloom_read_number(p, p_next - p, 3, 5);
        if (nal_unit_type == 5)
            chk += 1;
    }

    return chk;
}
