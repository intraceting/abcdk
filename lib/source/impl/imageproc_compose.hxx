/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_IMPL_IMAGEPROC_COMPOSE_HXX
#define ABCDK_IMPL_IMAGEPROC_COMPOSE_HXX

#include "general.hxx"

namespace abcdk
{
    namespace imageproc
    {
        template <typename T>
        ABCDK_INVOKE_DEVICE void compose_kernel(int channels, bool packed,
                                                T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                                T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                                T *scalar, size_t overlap_x, size_t overlap_y, size_t overlap_w, bool optimize_seam,
                                                size_t tid)
        {
            size_t y = tid / compose_w;
            size_t x = tid % compose_w;

            size_t panorama_offset[4] = {(size_t)-1, (size_t)-1, (size_t)-1, (size_t)-1};
            size_t compose_offset[4] = {(size_t)-1, (size_t)-1, (size_t)-1, (size_t)-1};
            size_t panorama_scalars = 0;
            size_t compose_scalars = 0;

            /*融合权重。-1.0 ～ 0～ 1.0 。*/
            double scale = 0;

            if (x >= compose_w || y >= compose_h)
                return;

            for (size_t i = 0; i < channels; i++)
            {
                panorama_offset[i] = abcdk::general::off<T>(packed, panorama_w, panorama_ws, panorama_h, channels, 0, x + overlap_x, y + overlap_y, i);
                compose_offset[i] = abcdk::general::off<T>(packed, compose_w, compose_ws, compose_h, channels, 0, x, y, i);

                /*计算融合图象素是否为填充色。*/
                panorama_scalars += (abcdk::general::obj<T>(panorama, panorama_offset[i]) == scalar[i] ? 1 : 0);

                /*计算融合图象素是否为填充色。*/
                compose_scalars += (abcdk::general::obj<T>(compose, compose_offset[i]) == scalar[i] ? 1 : 0);
            }

            if (panorama_scalars == channels)
            {
                /*全景图象素为填充色，只取融合图象素。*/
                scale = 0;
            }
            else if (compose_scalars == channels)
            {
                /*融合图象素为填充色，只取全景图象素。*/
                scale = 1;
            }
            else
            {
                /*判断是否在图像重叠区域，从左到右渐进分配融合重叠区域的顔色权重。*/
                if (x + overlap_x <= overlap_x + overlap_w)
                {
                    scale = ((overlap_w - x) / (double)overlap_w);
                    /*按需优化接缝线。*/
                    scale = (optimize_seam ? scale : 1 - scale);
                }
                else
                {
                    /*不在重叠区域，只取融合图象素。*/
                    scale = 0;
                }
            }

            for (size_t i = 0; i < channels; i++)
            {
                *abcdk::general::ptr<T>(panorama, panorama_offset[i]) = abcdk::general::blend<T>(abcdk::general::obj<T>(panorama, panorama_offset[i]), abcdk::general::obj<T>(compose, compose_offset[i]), scale);
            }
        }

        /**图像融合。*/
        template <typename T>
        ABCDK_INVOKE_HOST void compose(int channels, bool packed,
                                       T *panorama, size_t panorama_w, size_t panorama_ws, size_t panorama_h,
                                       T *compose, size_t compose_w, size_t compose_ws, size_t compose_h,
                                       T *stuff, size_t overlap_x, size_t overlap_y, size_t overlap_w, int optimize_seam)
        {

            for (size_t i = 0; i < compose_w * compose_h; i++)
            {
                compose_kernel<T>(channels, packed, panorama, panorama_w, panorama_ws, panorama_h,
                                  compose, compose_w, compose_ws, compose_h,
                                  stuff, overlap_x, overlap_y, overlap_w, optimize_seam, i);
            }
        }
    } // namespace imageproc

} // namespace abcdk

#endif // ABCDK_IMPL_IMAGEPROC_COMPOSE_HXX