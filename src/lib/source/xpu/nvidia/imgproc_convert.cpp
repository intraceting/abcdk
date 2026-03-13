/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"
#include "../common/imgproc.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace imgproc
        {
            static int _convert_cpu(image::metadata_t *dst, const image::metadata_t *src)
            {
                image::metadata_t *dst_cpu;
                image::metadata_t *src_cpu;
                int chk;

                dst_cpu = image::create(dst->width, dst->height, pixfmt::ffmpeg_to_local(dst->format), 16, 1);
                src_cpu = image::create(src->width, src->height, pixfmt::ffmpeg_to_local(src->format), 16, 1);

                if (!dst_cpu || !src_cpu)
                {
                    image::free(&dst_cpu);
                    image::free(&src_cpu);
                    return -ENOMEM;
                }

                image::copy(src, 0, src_cpu, 1);
                chk = common::imgproc::convert(src_cpu, dst_cpu);
                if (chk == 0)
                    image::copy(dst_cpu, 1, dst, 1);

                image::free(&dst_cpu);
                image::free(&src_cpu);
                return chk;
            }

            static int _convert_gpu(image::metadata_t *dst, const image::metadata_t *src)
            {
                image::metadata_t *tmp_dst = NULL;
                NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;

                NppiSize src_roi = {src->width, src->height};

                /* 颜色变换矩阵（标准 RGB → YUV 转换）*/
                Npp32f rgb_to_yuv_twist[3][4] = {
                    {0.299f, 0.587f, 0.114f, 0.0f},     // Y
                    {-0.169f, -0.331f, 0.500f, 128.0f}, // U
                    {0.500f, -0.419f, -0.081f, 128.0f}  // V
                };

                int chk;

                if (dst->format == AV_PIX_FMT_GRAY8)
                {
                    if (src->format == AV_PIX_FMT_RGB24)
                    {
                        npp_chk = nppiRGBToGray_8u_C3C1R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi);
                    }
                    else
                    {
                        tmp_dst = image::create(dst->width, dst->height, ABCDK_XPU_PIXFMT_RGB24, 16, 0);
                        if (!tmp_dst)
                            return -1;

                        chk = _convert_gpu(tmp_dst, src);
                        if (chk == 0)
                            chk = _convert_gpu(dst, tmp_dst);

                        image::free(&tmp_dst);
                        return chk;
                    }
                }
                else if (dst->format == AV_PIX_FMT_RGB24)
                {
                    if (src->format == AV_PIX_FMT_BGR24)
                    {
                        int dst_order[3] = {2, 1, 0};
                        npp_chk = nppiSwapChannels_8u_C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
                    }
                    else if (src->format == AV_PIX_FMT_RGB32)
                    {
                        int dst_order[3] = {0, 1, 2};
                        npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
                    }
                    else if (src->format == AV_PIX_FMT_BGR32)
                    {
                        int dst_order[3] = {2, 1, 0};
                        npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
                    }
                    else if (src->format == AV_PIX_FMT_YUV420P)
                    {
                        npp_chk = nppiYUV420ToRGB_8u_P3C3R(src->data, (int *)src->linesize, dst->data[0], dst->linesize[0], src_roi);
                    }
                    else if (src->format == AV_PIX_FMT_YUV422P)
                    {
                        npp_chk = nppiYUV422ToRGB_8u_P3C3R(src->data, (int *)src->linesize, dst->data[0], dst->linesize[0], src_roi);
                    }
                    else if (src->format == AV_PIX_FMT_YUV444P)
                    {
                        npp_chk = nppiYCbCr444ToRGB_JPEG_8u_P3C3R(src->data, src->linesize[0], dst->data[0], dst->linesize[0], src_roi);
                    }
                    else if (src->format == AV_PIX_FMT_NV12)
                    {
                        npp_chk = nppiNV12ToRGB_8u_P2C3R(src->data, src->linesize[0], dst->data[0], dst->linesize[0], src_roi);
                    }
                    else
                    {
                        return _convert_cpu(dst, src);
                    }
                }
                else if (dst->format == AV_PIX_FMT_BGR24)
                {
                    if (src->format == AV_PIX_FMT_RGB24)
                    {
                        int dst_order[3] = {2, 1, 0};
                        npp_chk = nppiSwapChannels_8u_C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
                    }
                    else if (src->format == AV_PIX_FMT_BGR32)
                    {
                        int dst_order[3] = {0, 1, 2};
                        npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
                    }
                    else if (src->format == AV_PIX_FMT_RGB32)
                    {
                        int dst_order[3] = {2, 1, 0};
                        npp_chk = nppiSwapChannels_8u_C4C3R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order);
                    }
                    else if (src->format == AV_PIX_FMT_YUV420P)
                    {
                        npp_chk = nppiYUV420ToBGR_8u_P3C3R(src->data, (int *)src->linesize, dst->data[0], dst->linesize[0], src_roi);
                    }
                    else if (src->format == AV_PIX_FMT_NV12)
                    {
                        npp_chk = nppiNV12ToBGR_8u_P2C3R(src->data, src->linesize[0], dst->data[0], dst->linesize[0], src_roi);
                    }
                    else
                    {
                        tmp_dst = image::create(dst->width, dst->height, ABCDK_XPU_PIXFMT_RGB24, 16, 0);
                        if (!tmp_dst)
                            return -1;

                        chk = _convert_gpu(tmp_dst, src);
                        if (chk == 0)
                            chk = _convert_gpu(dst, tmp_dst);

                        image::free(&tmp_dst);
                        return chk;
                    }
                }
                else if (dst->format == AV_PIX_FMT_YUV420P)
                {
                    if (src->format == AV_PIX_FMT_RGB24)
                    {
                        npp_chk = nppiRGBToYUV420_8u_C3P3R(src->data[0], src->linesize[0], dst->data, dst->linesize, src_roi);
                    }
                    else if (src->format == AV_PIX_FMT_NV12)
                    {
                        npp_chk = nppiNV12ToYUV420_8u_P2P3R(src->data, src->linesize[0], dst->data, dst->linesize, src_roi);
                    }
                    else
                    {
                        tmp_dst = image::create(dst->width, dst->height, ABCDK_XPU_PIXFMT_RGB24, 16, 0);
                        if (!tmp_dst)
                            return -1;

                        chk = _convert_gpu(tmp_dst, src);
                        if (chk == 0)
                            chk = _convert_gpu(dst, tmp_dst);

                        image::free(&tmp_dst);
                        return chk;
                    }
                }
                else if (dst->format == AV_PIX_FMT_YUV422P)
                {
                    if (src->format == AV_PIX_FMT_RGB24)
                    {
                        npp_chk = nppiRGBToYUV422_8u_C3P3R(src->data[0], src->linesize[0], dst->data, dst->linesize, src_roi);
                    }
                    else
                    {
                        tmp_dst = image::create(dst->width, dst->height, ABCDK_XPU_PIXFMT_RGB24, 16, 0);
                        if (!tmp_dst)
                            return -1;

                        chk = _convert_gpu(tmp_dst, src);
                        if (chk == 0)
                            chk = _convert_gpu(dst, tmp_dst);

                        image::free(&tmp_dst);
                        return chk;
                    }
                }
                else if (dst->format == AV_PIX_FMT_YUV444P)
                {
                    if (src->format == AV_PIX_FMT_RGB24)
                    {
                        npp_chk = nppiRGBToYCbCr444_JPEG_8u_C3P3R(src->data[0], src->linesize[0], dst->data, dst->linesize[0], src_roi);
                    }
                    else
                    {
                        tmp_dst = image::create(dst->width, dst->height, ABCDK_XPU_PIXFMT_RGB24, 16, 0);
                        if (!tmp_dst)
                            return -1;

                        chk = _convert_gpu(tmp_dst, src);
                        if (chk == 0)
                            chk = _convert_gpu(dst, tmp_dst);

                        image::free(&tmp_dst);
                        return chk;
                    }
                }
                else if (dst->format == AV_PIX_FMT_NV12)
                {
#if NPP_VERSION >= (12 * 1000 + 4 * 100 + 0)
                    if (src->format == AV_PIX_FMT_RGB24)
                    {
                        NppStreamContext stream_ctx = {0};
                        npp_chk = nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(src->data[0], src->linesize[0], dst->data, dst->linesize, src_roi, rgb_to_yuv_twist, stream_ctx);
                    }
                    else
#endif // #if NPP_VERSION >= (12 * 1000 + 4 * 100 + 0)
                    {
                        tmp_dst = image::create(dst->width, dst->height, ABCDK_XPU_PIXFMT_RGB24, 16, 0);
                        if (!tmp_dst)
                            return -1;

                        chk = _convert_gpu(tmp_dst, src);
                        if (chk == 0)
                            chk = _convert_gpu(dst, tmp_dst);

                        image::free(&tmp_dst);
                        return chk;
                    }
                }
                else if (dst->format == AV_PIX_FMT_RGB32)
                {
                    if (src->format == AV_PIX_FMT_RGB24)
                    {
                        int dst_order[4] = {0, 1, 2, 3};
                        npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order, 0);
                    }
                    else if (src->format == AV_PIX_FMT_BGR24)
                    {
                        int dst_order[4] = {2, 1, 0, 3};
                        npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order, 0);
                    }
                    else
                    {
                        tmp_dst = image::create(dst->width, dst->height, ABCDK_XPU_PIXFMT_RGB24, 16, 0);
                        if (!tmp_dst)
                            return -1;

                        chk = _convert_gpu(tmp_dst, src);
                        if (chk == 0)
                            chk = _convert_gpu(dst, tmp_dst);

                        image::free(&tmp_dst);
                        return chk;
                    }
                }
                else if (dst->format == AV_PIX_FMT_BGR32)
                {
                    if (src->format == AV_PIX_FMT_BGR24)
                    {
                        int dst_order[4] = {0, 1, 2, 3};
                        npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order, 0);
                    }
                    else if (src->format == AV_PIX_FMT_RGB24)
                    {
                        int dst_order[4] = {2, 1, 0, 3};
                        npp_chk = nppiSwapChannels_8u_C3C4R(src->data[0], src->linesize[0], dst->data[0], dst->linesize[0], src_roi, dst_order, 0);
                    }
                    else
                    {
                        tmp_dst = image::create(dst->width, dst->height, ABCDK_XPU_PIXFMT_RGB24, 16, 0);
                        if (!tmp_dst)
                            return -1;

                        chk = _convert_gpu(tmp_dst, src);
                        if (chk == 0)
                            chk = _convert_gpu(dst, tmp_dst);

                        image::free(&tmp_dst);
                        return chk;
                    }
                }
                else
                {
                    return _convert_cpu(dst, src);
                }

                if (npp_chk != NPP_SUCCESS && npp_chk != NPP_DOUBLE_SIZE_WARNING)
                    return -1;

                return 0;
            }

            int convert(const image::metadata_t *src, image::metadata_t *dst)
            {
                return _convert_gpu(dst, src);
            }

            int convert2(image::metadata_t **dst, abcdk_xpu_pixfmt_t pixfmt)
            {
                image::metadata_t *tmp_dst;
                image::metadata_t *dst_p;
                int chk;

                dst_p = *dst;

                if (pixfmt::ffmpeg_to_local(dst_p->format) == pixfmt)
                    return 0;

                tmp_dst = image::create(dst_p->width, dst_p->height, pixfmt, 16, 0);
                if (!tmp_dst)
                    return -1;

                chk = convert(dst_p, tmp_dst);
                if (chk != 0)
                {
                    image::free(&tmp_dst);
                    return -2;
                }

                image::free(dst);
                *dst = tmp_dst;
                
                return 0;
            }
        } // namespace imgproc
    } // namespace nvidia

} // namespace abcdk_xpu
