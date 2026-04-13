/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_UTIL_HXX
#define ABCDK_XPU_NVIDIA_UTIL_HXX

#include "abcdk/xpu/types.h"
#include "../base.in.h"

#ifdef __x86_64__
#include "ffnvcodec/dynlink_loader.h"
#include "ffnvcodec/dynlink_nvcuvid.h"
#endif // #ifdef __x86_64__

#ifdef __XPU_NVIDIA__MMAPI__
#include "v4l2_nv_extensions.h"
#endif //#ifdef __XPU_NVIDIA__MMAPI__

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace util
        {
            static inline void npp_get_context(NppStreamContext *npp_ctx, cudaStream_t stream = 0)
            {
                int dev_id;
                cudaGetDevice(&dev_id); 

                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, dev_id); 

                npp_ctx->hStream = stream;
                npp_ctx->nCudaDeviceId = dev_id;
                npp_ctx->nMultiProcessorCount = prop.multiProcessorCount;
                npp_ctx->nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
                npp_ctx->nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
                npp_ctx->nSharedMemPerBlock = prop.sharedMemPerBlock;
                
                cudaDeviceGetAttribute(&npp_ctx->nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, dev_id);
                cudaDeviceGetAttribute(&npp_ctx->nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, dev_id);

                if (stream != 0)
                    cudaStreamGetFlags(stream, &npp_ctx->nStreamFlags);
                else
                    npp_ctx->nStreamFlags = 0; // 默认流.

                npp_ctx->nReserved0 = 0; // 预留的.
            }

            static inline NppiInterpolationMode inter_local_to_nppi(abcdk_xpu_inter_t mode)
            {
                if (mode == ABCDK_XPU_INTER_NEAREST)
                    return NPPI_INTER_NN;
                else if (mode == ABCDK_XPU_INTER_LINEAR)
                    return NPPI_INTER_LINEAR;
                else if (mode == ABCDK_XPU_INTER_CUBIC)
                    return NPPI_INTER_CUBIC;
                else
                    return NPPI_INTER_UNDEFINED;
            }

#ifdef __x86_64__
            static inline cudaVideoCodec local_to_nvcodec(abcdk_xpu_vcodec_id_t id)
            {
                switch (id)
                {
                case ABCDK_XPU_VCODEC_ID_MJPEG:
                    return cudaVideoCodec_JPEG;
                case ABCDK_XPU_VCODEC_ID_MPEG1VIDEO:
                    return cudaVideoCodec_MPEG1;
                case ABCDK_XPU_VCODEC_ID_MPEG2VIDEO:
                    return cudaVideoCodec_MPEG2;
                case ABCDK_XPU_VCODEC_ID_MPEG4:
                    return cudaVideoCodec_MPEG4;
                case ABCDK_XPU_VCODEC_ID_H264:
                    return cudaVideoCodec_H264;
                case ABCDK_XPU_VCODEC_ID_HEVC:
                    return cudaVideoCodec_HEVC;
                case ABCDK_XPU_VCODEC_ID_VC1:
                    return cudaVideoCodec_VC1;
                case ABCDK_XPU_VCODEC_ID_VP8:
                    return cudaVideoCodec_VP8;
                case ABCDK_XPU_VCODEC_ID_VP9:
                    return cudaVideoCodec_VP9;
                default:
                    return cudaVideoCodec_NumCodecs;
                }
            }

            static inline cudaVideoCodec local_to_nvcodec(uint32_t id)
            {
                return local_to_nvcodec((abcdk_xpu_vcodec_id_t)id);
            }
#endif // #ifdef __x86_64__

#ifdef __XPU_NVIDIA__MMAPI__
            static inline int local_to_nvcodec(abcdk_xpu_vcodec_id_t id)
            {
                switch (id)
                {
                case ABCDK_XPU_VCODEC_ID_MPEG2VIDEO:
                    return -1;
                case ABCDK_XPU_VCODEC_ID_MPEG4:
                    return -1;
                case ABCDK_XPU_VCODEC_ID_H264:
                    return V4L2_PIX_FMT_H264;
                case ABCDK_XPU_VCODEC_ID_HEVC:
                    return V4L2_PIX_FMT_H265;
                case ABCDK_XPU_VCODEC_ID_VP8:
                    return -1;
                case ABCDK_XPU_VCODEC_ID_VP9:
                    return V4L2_PIX_FMT_VP9;
                default:
                    return -1;
                }
            }

            static inline int local_to_nvcodec(uint32_t id)
            {
                return local_to_nvcodec((abcdk_xpu_vcodec_id_t)id);
            }
#endif // #ifdef __XPU_NVIDIA__MMAPI__

        } // namespace util
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_UTIL_HXX