/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_UTIL_HXX
#define ABCDK_TORCH_UTIL_HXX

#include "invoke.hxx"

namespace abcdk
{
    namespace torch
    {
        namespace util
        {
            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T *ptr(void *data, size_t off = 0)
            {
                return (T *)(((uint8_t *)data) + off);
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T obj(void *data, size_t off = 0)
            {
                return *ptr<T>(data, off);
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE size_t off_nhwc(size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                size_t off = 0;

                off = n * h * ws;
                off += y * ws;
                off += x * c * sizeof(T);
                off += z * sizeof(T);

                return off;
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE size_t off_nchw(size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                size_t off = 0;

                off = n * c * h * ws;
                off += z * h * ws;
                off += y * ws;
                off += x * sizeof(T);

                return off;
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE size_t off_packed(size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                return off_nhwc<T>(w, ws, h, c, n, x, y, z);
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE size_t off_planar(size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                return off_nchw<T>(w, ws, h, c, n, x, y, z);
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE size_t off(bool packed, size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                if (packed)
                    return off_packed<T>(w, ws, h, c, n, x, y, z);
                else
                    return off_planar<T>(w, ws, h, c, n, x, y, z);
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T *ptr(void *data, bool packed, size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                return ptr<T>(data, off<T>(packed, w, ws, h, c, n, x, y, z));
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T obj(void *data, bool packed, size_t w, size_t ws, size_t h, size_t c, size_t n, size_t x, size_t y, size_t z)
            {
                return obj<T>(data, off<T>(packed, w, ws, h, c, n, x, y, z));
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE void swap(T &a, T &b)
            {
                T c;
                c = a;
                a = b;
                b = c;
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T min(T a, T b)
            {
                return (a < b ? a : b);
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T max(T a, T b)
            {
                return (a > b ? a : b);
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T clamp(T v, T a, T b)
            {
                return min(max(a, b), max(min(a, b), v));
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T blend(T a, T b, double scale)
            {
                return a * scale + b * (1 - scale);
            }

            template <typename T>
            ABCDK_TORCH_INVOKE_DEVICE T pixel(uint64_t src)
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

            /*线性坐标转NYXZ坐标。*/
            ABCDK_TORCH_INVOKE_DEVICE void idx2nyxz(size_t idx, size_t h, size_t w, size_t c, size_t &n, size_t &y, size_t &x, size_t &z)
            {
                n = idx / (h * w * c);
                y = (idx / (w * c)) % h;
                x = (idx / c) % w;
                z = idx % c;
            }

            /*断判点是否在线上。*/
            ABCDK_TORCH_INVOKE_DEVICE bool point_on_line(float x1, float y1, float x2, float y2, float px, float py, float linewidth)
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

        } // namespace util

    } // namespace torch

} // namespace abcdk

#endif // ABCDK_TORCH_UTIL_HXX