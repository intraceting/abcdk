/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/hevc.h"

void abcdk_hevc_extradata_clean(abcdk_hevc_extradata_t *extdata)
{
    assert(extdata != NULL);

    for(int i = 0;i<extdata->nal_array_num;i++)
        abcdk_object_unref(&extdata->nal_array[i].nal);

    memset(extdata,0,sizeof(*extdata));
}

ssize_t abcdk_hevc_extradata_serialize(const abcdk_hevc_extradata_t *extradata, void *data, size_t size)
{
    return -1;
}

void abcdk_hevc_extradata_deserialize(const void *data, size_t size, abcdk_hevc_extradata_t *extradata)
{
    abcdk_bit_t rbuf = {0};
    uint8_t d[3] = {0};
    const void *p1 = NULL, *p2 = NULL, *p3 = NULL;
    struct _nal_array *nal_p = NULL;

    assert(data != NULL && size > 0 && extradata != NULL);

    rbuf.data = (void *)data;
    rbuf.size = size;
    rbuf.pos = 0;

    d[0] = abcdk_bit_pread(&rbuf, 0, 8);
    d[1] = abcdk_bit_pread(&rbuf, 8, 8);
    d[2] = abcdk_bit_pread(&rbuf, 16, 8);

    /*see FFMPEG @ff_hevc_decode_extradata.*/
    if (size > 3 && (d[0] || d[1] || d[2] > 1))
    {

        extradata->version = abcdk_bit_read(&rbuf, 8);
        extradata->general_profile_space = abcdk_bit_read(&rbuf, 2);
        extradata->general_tier_flag = abcdk_bit_read(&rbuf, 1);
        extradata->general_profile_idc = abcdk_bit_read(&rbuf, 5);
        extradata->general_profile_compatibility_flags = abcdk_bit_read(&rbuf, 32);
        extradata->general_constraint_indicator_flags = abcdk_bit_read(&rbuf, 48);
        extradata->general_level_idc = abcdk_bit_read(&rbuf, 8);
        extradata->reserve1 = abcdk_bit_read(&rbuf, 4);
        extradata->min_spatial_segmentation_idc = abcdk_bit_read(&rbuf, 12);
        extradata->reserve2 = abcdk_bit_read(&rbuf, 6);
        extradata->parallelism_type = abcdk_bit_read(&rbuf, 2);
        extradata->reserve3 = abcdk_bit_read(&rbuf, 6);
        extradata->chroma_format = abcdk_bit_read(&rbuf, 2);
        extradata->reserve4 = abcdk_bit_read(&rbuf, 5);
        extradata->bit_depth_luma_minus8 = abcdk_bit_read(&rbuf, 3);
        extradata->reserve5 = abcdk_bit_read(&rbuf, 5);
        extradata->bit_depth_chroma_minus8 = abcdk_bit_read(&rbuf, 3);
        extradata->avg_frame_rate = abcdk_bit_read(&rbuf, 16);
        extradata->constant_frame_rate = abcdk_bit_read(&rbuf, 2);
        extradata->num_temporal_layers = abcdk_bit_read(&rbuf, 3);
        extradata->temporal_id_nested = abcdk_bit_read(&rbuf, 1);
        extradata->nal_length_size = abcdk_bit_read(&rbuf, 2);
        extradata->nal_array_num = abcdk_bit_read(&rbuf, 8);

        for (int i = 0; i < extradata->nal_array_num; i++)
        {
            nal_p = &extradata->nal_array[i];

            nal_p->array_completeness = abcdk_bit_read(&rbuf, 1);
            nal_p->reserve1 = abcdk_bit_read(&rbuf, 1);
            nal_p->unit_type = abcdk_bit_read(&rbuf, 6);
            nal_p->nal_num = abcdk_bit_read(&rbuf, 16);
            nal_p->nal = abcdk_object_alloc3(rbuf.size, nal_p->nal_num); // 足够的长度，保证不会溢出。

            for (int j = 0; j < nal_p->nal_num; j++)
            {
                nal_p->nal->sizes[j] = abcdk_bit_read(&rbuf, 16);
#if 0 
                memcpy(nal_p->nal->pptrs[j], ABCDK_PTR2VPTR(rbuf.data, rbuf.pos / 8), nal_p->nal->sizes[j]);
                rbuf.pos += (nal_p->nal->sizes[j] * 8);
#else
                abcdk_bit_read2buffer(&rbuf,nal_p->nal->pptrs[j], nal_p->nal->sizes[j]);
#endif 
            }
        }
    }
    else
    {
        p1 = rbuf.data;
        p2 = NULL;
        p3 = rbuf.data + rbuf.size - 1;/*末尾指针要减1。*/

        extradata->nal_array_num = 1;

        nal_p = &extradata->nal_array[0];

        nal_p->nal = abcdk_object_alloc3(rbuf.size, 5); // 足够的长度，保证不会溢出。

        for (int j = 0; j < 5; j++)
        {
            if (p1 > p3)
                return;

            p2 = abcdk_h2645_packet_split((void **)&p1, p3);
            if (p2 == NULL)
                return;

            nal_p->nal->sizes[j] = p1 - p2;
            memcpy(nal_p->nal->pptrs[j], p2, nal_p->nal->sizes[j]);

            nal_p->nal_num += 1;
        }
    }
}

int abcdk_hevc_irap(const void *data, size_t size)
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

        p = (void*)abcdk_h2645_packet_split(&p_next, p_end);
        if (!p)
            break;

        nal_unit_type = abcdk_bloom_read_number(p, p_next - p, 1, 6);
        if (nal_unit_type >= 16 && nal_unit_type <= 21)
            chk += 1;
    }

    return chk;
}
