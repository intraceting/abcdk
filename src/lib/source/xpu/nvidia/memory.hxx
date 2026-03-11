/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_MEMORY_HXX
#define ABCDK_XPU_NVIDIA_MEMORY_HXX

#include "abcdk/xpu/runtime.h"
#include "../base.in.h"
#include "../common/imgproc.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace memory
        {
            static inline void free(void *data, int in_host)
            {
                if (!data)
                    return;

                if (in_host)
                    ::free(data);
                else
                    cudaFree(data);
            }

            static inline void freep(void **data, int in_host)
            {
                void *data_p;

                if (!data || !*data)
                    return;

                data_p = *data;
                *data = NULL;

                free(data_p, in_host);
            }

            template <typename T>
            static inline T *alloc(size_t size, int in_host)
            {
                void *data;
                cudaError_t chk;

                assert(size > 0);

                if (in_host)
                {
                    data = malloc(size);
                }
                else
                {

                    chk = cudaMalloc(&data, size);
                    if (chk != cudaSuccess)
                        return NULL;
                }

                return (T *)data;
            }

            static inline int copy_1d(void *dst, int dst_in_host, const void *src, int src_in_host, size_t size)
            {
                cudaMemcpyKind kind = cudaMemcpyDefault;
                cudaError_t chk;

                assert(dst != NULL && src != NULL && size > 0);

                if (src_in_host && dst_in_host)
                    kind = cudaMemcpyHostToHost;
                else if (src_in_host)
                    kind = cudaMemcpyHostToDevice;
                else if (dst_in_host)
                    kind = cudaMemcpyDeviceToHost;
                else
                    kind = cudaMemcpyDeviceToDevice;

                chk = cudaMemcpy(dst, src, size, kind);

                if (chk != cudaSuccess)
                    return -1;

                return 0;
            }

            static inline int copy_2d(void *dst, size_t dst_pitch, size_t dst_x_bytes, size_t dst_y, int dst_in_host,
                                      const void *src, size_t src_pitch, size_t src_x_bytes, size_t src_y, int src_in_host,
                                      size_t roi_width_bytes, size_t roi_height)
            {
                CUDA_MEMCPY2D copy_args = {0};
                CUresult chk;

                assert(dst != NULL && src != NULL && roi_width_bytes > 0 && roi_height > 0);

                copy_args.dstXInBytes = dst_x_bytes;
                copy_args.dstY = dst_y;
                copy_args.dstMemoryType = (dst_in_host ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE);
                copy_args.dstHost = (dst_in_host ? dst : NULL);
                copy_args.dstDevice = (CUdeviceptr)(dst_in_host ? NULL : dst);
                copy_args.dstPitch = dst_pitch;

                copy_args.srcXInBytes = src_x_bytes;
                copy_args.srcY = src_y;
                copy_args.srcMemoryType = (src_in_host ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE);
                copy_args.srcHost = (src_in_host ? src : NULL);
                copy_args.srcDevice = (CUdeviceptr)(src_in_host ? NULL : src);
                copy_args.srcPitch = src_pitch;

                copy_args.WidthInBytes = roi_width_bytes;
                copy_args.Height = roi_height;

                chk = cuMemcpy2D(&copy_args);

                if (chk != CUDA_SUCCESS)
                    return -1;

                return 0;
            }

            void *cudaMemset(void *dst, int val, size_t size);

            static inline void *memset(void *dst, int val, size_t size, int in_host)
            {
                return (in_host ? ::memset(dst, val, size) : cudaMemset(dst, 0, size));
            }

            template <typename T>
            static inline T *alloc_z(size_t size, int in_host)
            {
                T *data;

                data = alloc<T>(size, in_host);
                if (!data)
                    return NULL;

                return (T *)memset(data, 0, size, in_host);
            }

            template <typename T>
            static inline T *clone(int dst_in_host, const T *src, size_t src_size, int src_in_host)
            {
                T *dst;
                int chk;

                dst = alloc<T>(src_size, dst_in_host);
                if (!dst)
                    return NULL;

                chk = copy_1d(dst, dst_in_host, src, src_in_host, src_size);
                if (chk == 0)
                    return dst;

                free(dst, dst_in_host);
                return NULL;
            }

        } // namespace memory
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_MEMORY_HXX