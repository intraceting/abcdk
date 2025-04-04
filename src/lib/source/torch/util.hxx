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
        } // namespace util

    } // namespace torch

} // namespace abcdk

#endif // ABCDK_TORCH_UTIL_HXX