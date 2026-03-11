/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_IMGPROC_COMPOSE_HXX
#define ABCDK_XPU_COMMON_IMGPROC_COMPOSE_HXX

#include "abcdk/xpu/imgproc.h"
#include "../base.in.h"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            /**
             * 全景图像融合(从左到右).
             *
             * @param [in out] panorama 全景图像.
             * @param [in] part 融合图像.
             * @param [in] scalar 填充色.
             * @param [in] overlap_x  融合图像在全景图像的左上角X坐标.
             * @param [in] overlap_y  融合图像在全景图像的左上角Y坐标.
             * @param [in] overlap_w  融合图像在全景图像中重叠宽度.
             * @param [in] optimize_seam 接缝美化. 0 禁用, !0 启用.
             *
             */
            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE void compose_kernel(int channels, bool packed,
                                                      T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                      T *part, size_t part_w, size_t part_ws, size_t part_h,
                                                      T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w,
                                                      bool optimize_seam, size_t tid)
            {

                size_t y = tid / part_w;
                size_t x = tid % part_w;

                size_t panorama_offset[4] = {(size_t)-1, (size_t)-1, (size_t)-1, (size_t)-1};
                size_t part_offset[4] = {(size_t)-1, (size_t)-1, (size_t)-1, (size_t)-1};
                size_t panorama_scalars = 0;
                size_t part_scalars = 0;

                /*融合权重: -1.0 ～ 0 ～ 1.0*/
                double scale = 0;

                if (x >= part_w || y >= part_h)
                    return;

                for (size_t i = 0; i < channels; i++)
                {
                    panorama_offset[i] = util::off<T>(packed, panorama_w, panorama_ws, panorama_h, channels, 0, x + overlap_x, y + overlap_y, i);
                    part_offset[i] = util::off<T>(packed, part_w, part_ws, part_h, channels, 0, x, y, i);

                    /*计算融合图象素是否为填充色.*/
                    panorama_scalars += (util::obj<T>(panorama, panorama_offset[i]) == util::pixel<T>(scalar[i]) ? 1 : 0);

                    /*计算融合图象素是否为填充色.*/
                    part_scalars += (util::obj<T>(part, part_offset[i]) == util::pixel<T>(scalar[i]) ? 1 : 0);
                }

                if (panorama_scalars == channels)
                {
                    /*全景图象素为填充色, 只取融合图象素.*/
                    scale = 0;
                }
                else if (part_scalars == channels)
                {
                    /*融合图象素为填充色, 只取全景图象素.*/
                    scale = 1;
                }
                else
                {
                    /*判断是否在图像重叠区域, 从左到右渐进分配融合重叠区域的顔色权重.*/
                    if (x + overlap_x <= overlap_x + overlap_w)
                    {
                        scale = ((overlap_w - x) / (double)overlap_w);
                        /*按需优化接缝线.*/
                        scale = (optimize_seam ? scale : 1 - scale);
                    }
                    else
                    {
                        /*不在重叠区域, 只取融合图象素.*/
                        scale = 0;
                    }
                }

                for (size_t i = 0; i < channels; i++)
                {
                    *util::ptr<T>(panorama, panorama_offset[i]) = util::blend<T>(util::obj<T>(panorama, panorama_offset[i]), util::obj<T>(part, part_offset[i]), scale);
                }
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_GLOBAL void compose_3d3d(int channels, bool packed,
                                                    T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                    T *part, size_t part_w, size_t part_ws, size_t part_h,
                                                    T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w,
                                                    bool optimize_seam, size_t tid = SIZE_MAX)
            {
#ifdef __NVCC__
                tid = util::kernel_thread_get_id();
#endif //__NVCC__

                compose_kernel<T>(channels, packed,
                                  panorama, panorama_w, panorama_ws, panorama_h,
                                  part, part_w, part_ws, part_h,
                                  scalar, overlap_x, overlap_y, overlap_w,
                                  optimize_seam, tid);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_HOST int compose(int pixfmt,
                                            T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                            T *part, size_t part_w, size_t part_ws, size_t part_h,
                                            T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w,
                                            bool optimize_seam)
            {
                int channel = abcdk_ffmpeg_pixfmt_get_channel((AVPixelFormat)pixfmt);

                if (channel <= 0)
                    return -1;

#ifdef __NVCC__
                dim3 grid, block;
                util::kernel_dim_make_3d3d(grid, block, part_w * part_h);

                compose_3d3d<T><<<grid, block>>>(channel, true, panorama, panorama_w, panorama_ws, panorama_h,
                                                 part, part_w, part_ws, part_h,
                                                 scalar, overlap_x, overlap_y, overlap_w,
                                                 optimize_seam);
#else //__NVCC__
                long cpus = sysconf(_SC_NPROCESSORS_ONLN);

#pragma omp parallel for num_threads(cpus)
                for (size_t tid = 0; tid < part_w * part_h; tid++)
                {
                    compose_3d3d<T>(channel, true, panorama, panorama_w, panorama_ws, panorama_h,
                                    part, part_w, part_ws, part_h,
                                    scalar, overlap_x, overlap_y, overlap_w,
                                    optimize_seam, tid);
                }
#endif //__NVCC__

                return 0;
            }

            __ABCDK_XPU_INVOKE_HOST int compose(AVFrame *panorama, const AVFrame *part,
                                            size_t overlap_x, size_t overlap_y, size_t overlap_w,
                                            const abcdk_xpu_scalar_t *scalar,
                                            int optimize_seam)
            {
                assert(panorama->format == AV_PIX_FMT_GRAY8 ||
                       panorama->format == AV_PIX_FMT_RGB24 ||
                       panorama->format == AV_PIX_FMT_BGR24 ||
                       panorama->format == AV_PIX_FMT_RGB32 ||
                       panorama->format == AV_PIX_FMT_BGR32);

                return compose<uint8_t>(panorama->format,
                                        (uint8_t *)panorama->data[0], panorama->width, panorama->linesize[0], panorama->height,
                                        (uint8_t *)part->data[0], part->width, part->linesize[0], part->height,
                                        (uint8_t *)scalar, overlap_x, overlap_y, overlap_w,
                                        optimize_seam);
            }

        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_IMGPROC_COM2POSE_HXX