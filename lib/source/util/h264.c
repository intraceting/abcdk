/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/h264.h"

ssize_t abcdk_h264_extradata_serialize(const abcdk_h264_extradata_t *extdata, void *data, size_t size)
{

}

void abcdk_h264_extradata_deserialize(const void *data, size_t size, abcdk_h264_extradata_t *extdata)
{
    abcdk_bit_t rbuf = {0};

    assert(data != NULL && size > 0 && extdata != NULL);

    rbuf.data = (void*)data;
    rbuf.size = size;
    rbuf.pos = 0;

    extdata->version = abcdk_bit_read(&rbuf,8);
    extdata->avc_profile = abcdk_bit_read(&rbuf,8);
    extdata->avc_compatibility = abcdk_bit_read(&rbuf,8);
    extdata->avc_level = abcdk_bit_read(&rbuf,8);
    extdata->reserve1 = abcdk_bit_read(&rbuf,6);
    extdata->nal_length_size = abcdk_bit_read(&rbuf,2);
    extdata->reserve2 = abcdk_bit_read(&rbuf,3);

    extdata->sps_num = abcdk_bit_read(&rbuf, 5);
    extdata->sps = abcdk_object_alloc3(rbuf.size, extdata->sps_num); //足够的长度，保证不会溢出。
    for (int i = 0; i < extdata->sps_num; i++)
    {
        extdata->sps->sizes[i] = abcdk_bit_read(&rbuf, 16);
        memcpy(extdata->sps->pptrs[i], ABCDK_PTR2VPTR(rbuf.data, rbuf.pos / 8), extdata->sps->sizes[i]);
        rbuf.pos += (extdata->sps->sizes[i] * 8);
    }

    extdata->pps_num = abcdk_bit_read(&rbuf, 8);
    extdata->pps = abcdk_object_alloc3(rbuf.size, extdata->sps_num); //足够的长度，保证不会溢出。
    for (int i = 0; i < extdata->pps_num; i++)
    {
        extdata->pps->sizes[i] = abcdk_bit_read(&rbuf, 16);
        memcpy(extdata->pps->pptrs[i], ABCDK_PTR2VPTR(rbuf.data, rbuf.pos / 8), extdata->pps->sizes[i]);
        rbuf.pos += (extdata->pps->sizes[i] * 8);
    }
}