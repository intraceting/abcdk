/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/cuda/avutil.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

int abcdk_cuda_avframe_compose(AVFrame *panorama, const AVFrame *compose,
                               uint8_t scalar[4], size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
{
    AVFrame *tmp_panorama = NULL, *tmp_compose = NULL;
    int panorama_in_host, compose_in_host;
    int chk;

    assert(panorama != NULL && compose != NULL && scalar != NULL && overlap_w > 0);

    assert(panorama->format == compose->format);

    assert(panorama->format == (int)AV_PIX_FMT_GRAY8 ||
           panorama->format == (int)AV_PIX_FMT_RGB24 || panorama->format == (int)AV_PIX_FMT_BGR24 ||
           panorama->format == (int)AV_PIX_FMT_RGB32 || panorama->format == (int)AV_PIX_FMT_BGR32);

    panorama_in_host = (abcdk_cuda_avframe_memory_type(panorama) != CU_MEMORYTYPE_DEVICE);
    compose_in_host = (abcdk_cuda_avframe_memory_type(compose) != CU_MEMORYTYPE_DEVICE);

    if (compose_in_host)
    {
        tmp_compose = abcdk_cuda_avframe_clone(0, compose);
        if (!tmp_compose)
            return -1;

        chk = abcdk_cuda_avframe_compose(panorama, tmp_compose, scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
        av_frame_free(&tmp_compose);

        return chk;
    }

    /*最后检查这个参数，因为输出项需要复制。*/
    if (panorama_in_host)
    {
        tmp_panorama = abcdk_cuda_avframe_clone(0, panorama);
        if (!tmp_panorama)
            return -1;

        chk = abcdk_cuda_avframe_remap(tmp_panorama, compose, scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
        if (chk == 0)
            abcdk_cuda_avframe_copy(dst, tmp_panorama);
        av_frame_free(&tmp_panorama);

        return chk;
    }

    if (panorama->format == AV_PIX_FMT_GRAY8)
    {
        chk = abcdk_cuda_imgproc_compose_8u_C1R(panorama->data[0], panorama->width, panorama->linesize[0], panorama->height,
                                                compose->data[0], compose->width, compose->linesize[0], compose->height,
                                                scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
    }
    else if (panorama->format == (int)AV_PIX_FMT_RGB24 || panorama->format == (int)AV_PIX_FMT_BGR24)
    {
        chk = abcdk_cuda_imgproc_compose_8u_C3R(panorama->data[0], panorama->width, panorama->linesize[0], panorama->height,
                                                compose->data[0], compose->width, compose->linesize[0], compose->height,
                                                scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
    }
    else if (panorama->format == (int)AV_PIX_FMT_RGB32 || panorama->format == (int)AV_PIX_FMT_BGR32)
    {
        chk = abcdk_cuda_imgproc_compose_8u_C4R(panorama->data[0], panorama->width, panorama->linesize[0], panorama->height,
                                                compose->data[0], compose->width, compose->linesize[0], compose->height,
                                                scalar, overlap_x, overlap_y, overlap_w, optimize_seam);
    }
    else
    {
        chk = -1;
    }

    if (chk != 0)
        return -1;

    return 0;
}

#endif // AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__