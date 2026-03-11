/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_UTIL_HXX
#define ABCDK_XPU_COMMON_UTIL_HXX

#include "../base.in.h"

namespace abcdk_xpu
{
    namespace common
    {
        namespace util
        {
            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T *ptr(void *data, size_t off = 0)
            {
                return (T *)(((uint8_t *)data) + off);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T obj(void *data, size_t off = 0)
            {
                return *ptr<T>(data, off);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE size_t off_nhwc(size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                size_t off = 0;

                off = n * h * ws;
                off += y * ws;
                off += x * c * sizeof(T);
                off += z * sizeof(T);

                return off;
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE size_t off_nchw(size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                size_t off = 0;

                off = n * c * h * ws;
                off += z * h * ws;
                off += y * ws;
                off += x * sizeof(T);

                return off;
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE size_t off_packed(size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                return off_nhwc<T>(w, ws, h, c, n, x, y, z);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE size_t off_planar(size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                return off_nchw<T>(w, ws, h, c, n, x, y, z);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE size_t off(bool packed, size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                if (packed)
                    return off_packed<T>(w, ws, h, c, n, x, y, z);
                else
                    return off_planar<T>(w, ws, h, c, n, x, y, z);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T *ptr(void *data, bool packed, size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                return ptr<T>(data, off<T>(packed, w, ws, h, c, n, x, y, z));
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T obj(void *data, bool packed, size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                return obj<T>(data, off<T>(packed, w, ws, h, c, n, x, y, z));
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE void swap(T &a, T &b)
            {
                T c;
                c = a;
                a = b;
                b = c;
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T min(T a, T b)
            {
                return (a < b ? a : b);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T max(T a, T b)
            {
                return (a > b ? a : b);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T clamp(T v, T a, T b)
            {
                if (v < a)
                    return a;
                if (v > b)
                    return b;
                return v;
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T blend(T a, T b, double scale)
            {
                return a * scale + b * (1 - scale);
            }

            template <typename T>
            __ABCDK_XPU_INVOKE_DEVICE T pixel(uint64_t src)
            {
                if (sizeof(T) == sizeof(uint8_t))
                    return (T)clamp<uint64_t>(src, 0, 0xff);
                else if (sizeof(T) == sizeof(uint16_t))
                    return (T)clamp<uint64_t>(src, 0, 0xffff);
                else if (sizeof(T) == sizeof(uint32_t))
                    return (T)clamp<uint64_t>(src, 0, 0xffffffff);
                else
                    return (T)src;
            }

            /*线性坐标转NYXZ坐标.*/
            __ABCDK_XPU_INVOKE_DEVICE void idx2nyxz(size_t idx, size_t h, size_t w, size_t c, size_t &n, size_t &y, size_t &x, size_t &z)
            {
                n = idx / (h * w * c);
                y = (idx / (w * c)) % h;
                x = (idx / c) % w;
                z = idx % c;
            }

            /*断判点是否在线上.*/
            __ABCDK_XPU_INVOKE_DEVICE bool point_on_line(float x1, float y1, float x2, float y2, float px, float py, float linewidth)
            {
                float vx = x2 - x1;
                float vy = y2 - y1;
                float ux = px - x1;
                float uy = py - y1;

                float v_len2 = vx * vx + vy * vy;
                if (v_len2 == 0)
                    return false; // 线段退化为点

                float t = (ux * vx + uy * vy) / v_len2;

                if (t < 0.0f || t > 1.0f)
                    return false; // 投影不在线段内

                // 计算投影点坐标
                float proj_x = x1 + t * vx;
                float proj_y = y1 + t * vy;

                // 计算距离
                float dx = px - proj_x;
                float dy = py - proj_y;
                float dist = sqrtf(dx * dx + dy * dy);

                return dist <= linewidth / 2.0f;
            }

#ifdef __NVCC__
            __ABCDK_XPU_INVOKE_DEVICE size_t kernel_thread_get_id()
            {
                // 线程在 block 内线性编号
                size_t thread_in_block = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

                // block 在 grid 内线性编号
                size_t block_offset = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z);

                // 全局线程编号
                return thread_in_block + block_offset;
            }

            __ABCDK_XPU_INVOKE_HOST void make_dim_make_3d(dim3 &dim, size_t n, size_t b)
            {
                size_t k;
                unsigned int x, y;

                assert(n > 0 && b > 0);

                k = (n - 1) / b + 1;
                x = k;
                y = 1;

                if (x > 65535)
                {
                    x = ceil(sqrt(k));
                    y = (n - 1) / (x * b) + 1;
                }

                dim.x = x;
                dim.y = y;
                dim.z = 1;
            }

            __ABCDK_XPU_INVOKE_HOST void kernel_dim_make_3d3d(dim3 &grid, dim3 &block, size_t n)
            {
                make_dim_make_3d(block, 8, 8);
                make_dim_make_3d(grid, n, block.x * block.y * block.z);
            }

#endif // #ifdef __NVCC__

            __ABCDK_XPU_INVOKE_HOST cv::Mat AVFrame2cvMat(const AVFrame *src)
            {
                assert(src->format == AV_PIX_FMT_GRAY8 ||
                       src->format == AV_PIX_FMT_RGB24 ||
                       src->format == AV_PIX_FMT_BGR24 ||
                       src->format == AV_PIX_FMT_RGB32 ||
                       src->format == AV_PIX_FMT_BGR32 ||
                       src->format == AV_PIX_FMT_GRAYF32);

                if (src->format == AV_PIX_FMT_GRAYF32)
                {
                    return cv::Mat(src->height, src->width, CV_32FC(1), (void *)src->data[0], src->linesize[0]);
                }
                else if (src->format == AV_PIX_FMT_RGB32 || src->format == AV_PIX_FMT_BGR32)
                {
                    return cv::Mat(src->height, src->width, CV_8UC(4), (void *)src->data[0], src->linesize[0]);
                }
                else if (src->format == AV_PIX_FMT_RGB24 || src->format == AV_PIX_FMT_BGR24)
                {
                    return cv::Mat(src->height, src->width, CV_8UC(3), (void *)src->data[0], src->linesize[0]);
                }
                else if (src->format == AV_PIX_FMT_GRAY8)
                {
                    return cv::Mat(src->height, src->width, CV_8UC(1), (void *)src->data[0], src->linesize[0]);
                }

                return cv::Mat();//Nooooooooooooooooooooooooooo.
            }

            template <typename T>
            static inline void delete_object(T **ctx)
            {
                T *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                delete ctx_p;
            }

            template <typename T>
            static inline void delete_array(T **ctx)
            {
                T *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                delete[] ctx_p;
            }
        } // namespace util
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_UTIL_HXX