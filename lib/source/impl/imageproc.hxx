/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_IMPL_IMAGEPROC_HXX
#define ABCDK_IMPL_IMAGEPROC_HXX

#include "general.hxx"

namespace abcdk
{
    namespace imageproc
    {
        template <typename T>
        ABCDK_INVOKE_DEVICE void brightness_kernel(int channels, bool packed,
                                                   T *dst, size_t dst_ws, T *src, size_t src_ws,
                                                   size_t w, size_t h, float *alpha, float *bate,
                                                   size_t tid)
        {
            size_t y = tid / w;
            size_t x = tid % w;

            if (x >= w || y >= h)
                return;

            for (size_t z = 0; z < channels; z++)
            {
                size_t src_offset = abcdk::general::off<T>(packed, w, src_ws, h, channels, 0, x, y, z);
                size_t dst_offset = abcdk::general::off<T>(packed, w, dst_ws, h, channels, 0, x, y, z);

                dst[dst_offset] = (T)abcdk::general::pixel_clamp<float>(src[src_offset] * alpha[z] + bate[z]);
            }
        }

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
                panorama_scalars += (panorama[panorama_offset[i]] == scalar[i] ? 1 : 0);

                /*计算融合图象素是否为填充色。*/
                compose_scalars += (compose[compose_offset[i]] == scalar[i] ? 1 : 0);
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
                panorama[panorama_offset[i]] = abcdk::general::blend<T>(panorama[panorama_offset[i]], compose[compose_offset[i]], scale);
            }
        }

        template <typename T>
        ABCDK_INVOKE_DEVICE void defog_kernel(int channels, bool packed,
                                              T *dst, size_t dst_ws, T *src, size_t src_ws,
                                              size_t w, size_t h, float dack_m, T dack_a, float dack_w,
                                              size_t tid)
        {

            size_t y = tid / w;
            size_t x = tid % w;

            if (x >= w || y >= h)
                return;

            T dack_c = (T)abcdk::general::pixel_clamp<uint32_t>(0xffffffff);
            size_t src_of[4] = {0, 0, 0, 0};
            size_t dst_of[4] = {0, 0, 0, 0};

            for (size_t z = 0; z < channels; z++)
            {
                src_of[z] = abcdk::general::off<T>(packed, w, src_ws, h, channels, 0, x, y, z);
                dst_of[z] = abcdk::general::off<T>(packed, w, dst_ws, h, channels, 0, x, y, z);

                if (dack_c > src[src_of[z]])
                    dack_c = src[src_of[z]];
            }

            float t = abcdk::general::max<float>(dack_m, (1.0 - dack_w / dack_a * dack_c));

            for (size_t z = 0; z < channels; z++)
            {
                dst[dst_of[z]] = abcdk::general::pixel_clamp<T>(((src[src_of[z]] - dack_a) / t + dack_a));
            }
        }

        template <typename T>
        ABCDK_INVOKE_DEVICE void drawrect_kernel(int channels, bool packed,
                                                 T *dst, size_t w, size_t ws, size_t h,
                                                 T *color, int weight, int *corner,
                                                 size_t tid)
        {
            size_t y = tid / w;
            size_t x = tid % w;

            if (x >= w || y >= h)
                return;

            int x1 = corner[0];
            int y1 = corner[1];
            int x2 = corner[2];
            int y2 = corner[3];

            int chk = 0x00;

            /*上边*/
            if (x >= x1 && y >= y1 && x <= x2 && y <= y1 + weight && y <= y2)
                chk |= 0x01;

            /*下边*/
            if (x >= x1 && y <= y2 && x <= x2 && y >= y2 - weight && y >= y1)
                chk |= 0x02;

            /*左边*/
            if (x >= x1 && y >= y1 && x <= x1 + weight && y <= y2 && x <= x2)
                chk |= 0x04;

            /*右边*/
            if (x >= x2 - weight && y >= y1 && x <= x2 && y <= y2 && x >= x1)
                chk |= 0x08;

            /*为0表示不需要填充颜色。*/
            if (chk == 0)
                return;

            /*填充颜色。*/
            for (size_t z = 0; z < channels; z++)
            {
                size_t off = abcdk::general::off<T>(packed, w, ws, h, channels, 0, x, y, z);
                dst[off] = color[z];
            }
        }

        template <typename T>
        ABCDK_INVOKE_DEVICE void stuff_kernel(int channels, bool packed, T *dst, size_t width, size_t pitch, size_t height, T *scalar, size_t tid)
        {

            size_t y = tid / width;
            size_t x = tid % width;

            if (x >= width || y >= height)
                return;

            for (size_t i = 0; i < channels; i++)
            {
                size_t offset = abcdk::general::off<T>(packed, width, pitch, height, channels, 0, x, y, i);

                dst[offset] = (scalar ? scalar[i] : (T)0);
            }
        }

    } // namespace imageproc

} // namespace abcdk

#endif // ABCDK_IMPL_IMAGEPROC_HXX