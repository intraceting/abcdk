/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"


#ifdef __cuda_cuda_h__

int abcdk_cuda_avframe_save(const char *dst, const AVFrame *src)
{
    AVFrame *tmp_src = NULL;
    int src_bit_depth;
    int src_in_host;
    int chk;

    assert(dst != NULL && src != NULL);

    /*
     * BMP格式使用BGR24或BGR32。
     * 这里统一转成BRG32。
    */
    
    if (src->format != (int)AV_PIX_FMT_BGR32)
    {
        tmp_src = abcdk_avframe_alloc(src->width, src->height, AV_PIX_FMT_BGR32, 4); // BMP格式要求行以4字节对齐。
        if (!tmp_src)
            return -1;

        chk = abcdk_cuda_avframe_convert(tmp_src,src);//转换格式。

        if(chk == 0)
            chk = abcdk_cuda_avframe_save(dst, tmp_src);

        av_frame_free(&tmp_src);

        return chk;
    }

    src_in_host = (abcdk_cuda_avframe_memory_type(src) != CU_MEMORYTYPE_DEVICE);
    
    if (!src_in_host)
    {
        tmp_src = abcdk_cuda_avframe_clone(1, src);
        if (!tmp_src)
            return -1;

        chk = abcdk_cuda_avframe_save(dst, tmp_src);
        av_frame_free(&tmp_src);

        return chk;
    }

    src_bit_depth = abcdk_avimage_pixfmt_bits((enum AVPixelFormat)src->format, 0);

    /*BMP图像默认是倒投影存储。这里高度传入负值，使图像正投影存储。*/
    chk = abcdk_bmp_save_file(dst, src->data[0], src->linesize[0], src->width, -src->height, src_bit_depth);
    if (chk != 0)
        return -1;

    return 0;
}

#endif //__cuda_cuda_h__