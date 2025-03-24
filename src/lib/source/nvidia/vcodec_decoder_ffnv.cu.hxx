/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_VCODEC_DECODER_FFNV_HXX
#define ABCDK_NVIDIA_VCODEC_DECODER_FFNV_HXX

#include "abcdk/util/queue.h"
#include "abcdk/torch/vcodec.h"
#include "abcdk/nvidia/nvidia.h"
#include "abcdk/nvidia/image.h"
#include "abcdk/nvidia/context.h"
#include "vcodec_decoder.cu.hxx"
#include "vcodec_util.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef FFNV_CUDA_DYNLINK_LOADER_H


namespace abcdk
{
    namespace cuda
    {
        namespace vcodec
        {
            class decoder_ffnv : public decoder
            {
            public:
                static decoder *create(CUcontext cuda_ctx)
                {
                    decoder *ctx = new decoder_ffnv(cuda_ctx);
                    if (!ctx)
                        return NULL;

                    return ctx;
                }

                static void destory(decoder **ctx)
                {
                    decoder *ctx_p;

                    if (!ctx || !*ctx)
                        return;

                    ctx_p = *ctx;
                    *ctx = NULL;

                    delete (decoder_ffnv *)ctx_p;
                }

                static unsigned long get_decode_surfaces(cudaVideoCodec eCodec, unsigned int nWidth, unsigned int nHeight)
                {
                    if (eCodec == cudaVideoCodec_VP9)
                    {
                        return 12;
                    }

                    if (eCodec == cudaVideoCodec_H264 || eCodec == cudaVideoCodec_H264_SVC || eCodec == cudaVideoCodec_H264_MVC)
                    {
                        // assume worst-case of 20 decode surfaces for H264
                        return 20;
                    }

                    if (eCodec == cudaVideoCodec_HEVC)
                    {
                        // ref HEVC spec: A.4.1 General tier and level limits
                        // currently assuming level 6.2, 8Kx4K
                        int MaxLumaPS = 35651584;
                        int MaxDpbPicBuf = 6;
                        int PicSizeInSamplesY = (int)(nWidth * nHeight);
                        int MaxDpbSize;
                        if (PicSizeInSamplesY <= (MaxLumaPS >> 2))
                            MaxDpbSize = MaxDpbPicBuf * 4;
                        else if (PicSizeInSamplesY <= (MaxLumaPS >> 1))
                            MaxDpbSize = MaxDpbPicBuf * 2;
                        else if (PicSizeInSamplesY <= ((3 * MaxLumaPS) >> 2))
                            MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
                        else
                            MaxDpbSize = MaxDpbPicBuf;
                        return ABCDK_MIN(MaxDpbSize, 16) + 4;
                    }

                    return 8;
                }

                static int CUDAAPI video_sequence_cb(void *userdata, CUVIDEOFORMAT *format)
                {
                    return ((decoder_ffnv *)userdata)->video_sequence(format);
                }

                static int CUDAAPI picture_decode_cb(void *userdata, CUVIDPICPARAMS *params)
                {
                    return ((decoder_ffnv *)userdata)->picture_decode(params);
                }

                static int CUDAAPI picture_display_cb(void *userdata, CUVIDPARSERDISPINFO *info)
                {
                    return ((decoder_ffnv *)userdata)->picture_display(info);
                }

                static void frame_queue_destroy_cb(void *msg)
                {
                    abcdk_torch_frame_free((abcdk_torch_frame_t **)&msg);
                }

            private:
                CuvidFunctions *m_funcs;

                CUcontext m_gpu_ctx;
                CUvideoctxlock m_ctx_lock;

                CUVIDEOFORMATEX m_ext_data;

                CUvideoparser m_parser;
                CUvideodecoder m_decoder;

                CUVIDEOFORMAT m_videoformat;
                int m_pic_in_decode_order[32];
                int m_decode_pic_count;

                abcdk_queue_t *m_frame_queue;

            public:
                decoder_ffnv(CUcontext cuda_ctx)
                {
                    m_funcs = NULL;
                    cuvid_load_functions(&m_funcs, NULL);

                    m_gpu_ctx = cuda_ctx;
                    m_ctx_lock = NULL;
                    m_parser = NULL;
                    m_decoder = NULL;

                    memset(&m_videoformat, 0, sizeof(CUVIDEOFORMAT));

                    memset(&m_pic_in_decode_order[0], 0, sizeof(int) * 32);
                    m_decode_pic_count = 0;

                    m_frame_queue = NULL;
                }

                virtual ~decoder_ffnv()
                {
                    close();

                    cuvid_free_functions(&m_funcs);
                }

            protected:
                int video_sequence(CUVIDEOFORMAT *format)
                {
                    CUVIDDECODECAPS decode_caps;
                    CUVIDDECODECREATEINFO decode_info;
                    int decode_surface;
                    CUresult chk;

                    memset(&decode_caps, 0, sizeof(decode_caps));
                    memset(&decode_info, 0, sizeof(decode_info));

                    /*Copy*/
                    m_videoformat = *format;

                    decode_caps.eCodecType = m_videoformat.codec;
                    decode_caps.eChromaFormat = m_videoformat.chroma_format;
                    decode_caps.nBitDepthMinus8 = m_videoformat.bit_depth_luma_minus8;

                    cuCtxPushCurrent(m_gpu_ctx);
                    chk = m_funcs->cuvidGetDecoderCaps(&decode_caps);
                    cuCtxPopCurrent(NULL);

                    /*也许不支持*/
                    if ((chk != CUDA_SUCCESS) || !decode_caps.bIsSupported)
                        return 0;

                    decode_surface = get_decode_surfaces(m_videoformat.codec, m_videoformat.coded_width, m_videoformat.coded_height);
                    
                    decode_info.CodecType = m_videoformat.codec;
                    decode_info.ChromaFormat = m_videoformat.chroma_format;

                    decode_info.bitDepthMinus8 = decode_caps.nBitDepthMinus8;

                    if (decode_caps.eChromaFormat == cudaVideoChromaFormat_420 || decode_caps.eChromaFormat == cudaVideoChromaFormat_Monochrome)
                        decode_info.OutputFormat = decode_info.bitDepthMinus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
                    else if (decode_caps.eChromaFormat == cudaVideoChromaFormat_444)
                        decode_info.OutputFormat = decode_info.bitDepthMinus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
                    
                  
                    decode_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
                    decode_info.ulNumOutputSurfaces = 2; //decode_surface;
                    decode_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
                    decode_info.ulNumDecodeSurfaces = decode_surface;
                    decode_info.vidLock = m_ctx_lock;
                    decode_info.ulWidth = m_videoformat.coded_width;
                    decode_info.ulHeight = m_videoformat.coded_height;
                    decode_info.ulMaxWidth = m_videoformat.coded_width;
                    decode_info.ulMaxHeight = m_videoformat.coded_height;
                    decode_info.display_area.top = m_videoformat.display_area.top;
                    decode_info.display_area.bottom = m_videoformat.display_area.bottom;
                    decode_info.display_area.left = m_videoformat.display_area.left;
                    decode_info.display_area.right = m_videoformat.display_area.right;
                    decode_info.ulTargetWidth = abcdk_align(m_videoformat.display_area.right - m_videoformat.display_area.left, 2);
                    decode_info.ulTargetHeight = abcdk_align(m_videoformat.display_area.bottom - m_videoformat.display_area.top, 2);
                    decode_info.target_rect.top = 0;
                    decode_info.target_rect.bottom = decode_info.ulTargetHeight;
                    decode_info.target_rect.left = 0;
                    decode_info.target_rect.right = decode_info.ulTargetWidth;

                    cuCtxPushCurrent(m_gpu_ctx);
                    chk = m_funcs->cuvidCreateDecoder(&m_decoder, &decode_info);
                    cuCtxPopCurrent(NULL);

                    if(chk != CUDA_SUCCESS)
                        return 0;

                    return decode_surface;
                }

                int picture_decode(CUVIDPICPARAMS *params)
                {
                    CUresult chk;

                    if (!m_decoder)
                        return 0;

                    m_pic_in_decode_order[params->CurrPicIdx] = m_decode_pic_count++;

                    cuCtxPushCurrent(m_gpu_ctx);
                    chk = m_funcs->cuvidDecodePicture(m_decoder, params);
                    cuCtxPopCurrent(NULL);

                    if(chk != CUDA_SUCCESS)
                        return 0;

                    return 1;
                }

                int picture_display(CUVIDPARSERDISPINFO *info)
                {
                    CUVIDPROCPARAMS params;
                    CUVIDGETDECODESTATUS status;
                    CUdeviceptr src_frame = 0;
                    unsigned int src_pitch = 0;
                    abcdk_torch_image_t tmp_src;
                    abcdk_torch_frame_t *frame_src;
                    CUresult cuda_chk;
                    int chk;

                    if (!m_decoder)
                        return 0;

                    memset(&params,0,sizeof(params));
                    memset(&status,0,sizeof(status));

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    params.progressive_frame = info->progressive_frame;
                    params.second_field = info->repeat_first_field + 1;
                    params.top_field_first = info->top_field_first;
                    params.unpaired_field = info->repeat_first_field < 0;
                    params.output_stream = 0;

                    cuda_chk = m_funcs->cuvidMapVideoFrame(m_decoder, info->picture_index, &src_frame, &src_pitch, &params);
                    if (cuda_chk == CUDA_SUCCESS)
                    {
                        cuda_chk = m_funcs->cuvidGetDecodeStatus(m_decoder, info->picture_index, &status);
                        if (cuda_chk == CUDA_SUCCESS && (status.decodeStatus == cuvidDecodeStatus_Error || status.decodeStatus == cuvidDecodeStatus_Error_Concealed))
                        {
                            abcdk_trace_printf(LOG_WARNING, TT("Decode Error occurred for picture %d"), m_pic_in_decode_order[info->picture_index]);
                        }
                    }

                    tmp_src.tag = ABCDK_TORCH_TAG_HOST;
                    tmp_src.width = m_videoformat.display_area.right - m_videoformat.display_area.left;
                    tmp_src.height = m_videoformat.display_area.bottom - m_videoformat.display_area.top;
                    tmp_src.pixfmt = ABCDK_TORCH_PIXFMT_NV12;

                    tmp_src.stride[0] = src_pitch;
                    tmp_src.stride[1] = src_pitch;
                    tmp_src.stride[2] = tmp_src.stride[3] = 0;

                    abcdk_torch_imgutil_fill_pointer(tmp_src.data, tmp_src.stride, tmp_src.height, tmp_src.pixfmt, (void *)src_frame);

                    frame_src = abcdk_cuda_frame_create(tmp_src.width, tmp_src.height, tmp_src.pixfmt, 1);
                    if (frame_src)
                    {
                        abcdk_cuda_image_copy(frame_src->img, &tmp_src);

                        frame_src->pts = info->timestamp; // bind PTS

                        abcdk_queue_lock(m_frame_queue);
                        chk = abcdk_queue_push(m_frame_queue, frame_src);
                        abcdk_queue_unlock(m_frame_queue);

                        /*加入队列失败，直接删除。*/
                        if (chk != 0)
                            abcdk_torch_frame_free(&frame_src);
                    }
                    else
                    {
                        abcdk_trace_printf(LOG_WARNING, TT("内存不足。"));
                    }

                    chk = m_funcs->cuvidUnmapVideoFrame(m_decoder, src_frame);
                    assert(chk == CUDA_SUCCESS);

                    return 1;
                }

            public:
                virtual void close()
                {
                    if (!m_funcs)
                        return;

                    if (m_gpu_ctx)
                        cuCtxPushCurrent(m_gpu_ctx);

                    if (m_parser)
                        m_funcs->cuvidDestroyVideoParser(m_parser);
                    m_parser = NULL;

                    if (m_decoder)
                        m_funcs->cuvidDestroyDecoder(m_decoder);
                    m_decoder = NULL;

                    if (m_ctx_lock)
                        m_funcs->cuvidCtxLockDestroy(m_ctx_lock);
                    m_ctx_lock = NULL;

                    /*在这里删除。*/
                    abcdk_queue_free(&m_frame_queue);

                    if (m_gpu_ctx)
                        cuCtxPopCurrent(NULL);
                }

                virtual int open(abcdk_torch_vcodec_param_t *param)
                {
                    CUVIDPARSERPARAMS params;
                    CUresult chk;

                    assert(param != NULL);

                    if (!m_funcs)
                        return -1;

                    m_frame_queue = abcdk_queue_alloc(frame_queue_destroy_cb);
                    if (!m_frame_queue)
                        return -1;

                    chk = m_funcs->cuvidCtxLockCreate(&m_ctx_lock, m_gpu_ctx);
                    if (chk != CUDA_SUCCESS)
                        return -1;

                    memset(&params, 0, sizeof(params));
                    
                    params.CodecType = (cudaVideoCodec)vcodec_to_nvcodec(param->format);
                    params.ulMaxNumDecodeSurfaces = 25;
                    params.ulMaxDisplayDelay = 4;
                    params.pUserData = this;
                    params.pfnSequenceCallback = video_sequence_cb;
                    params.pfnDecodePicture = picture_decode_cb;
                    params.pfnDisplayPicture = picture_display_cb;
                    params.pExtVideoInfo = NULL;

                    if (param->ext_data != NULL && param->ext_size > 0)
                    {
                        /*空间有限。*/
                        if (sizeof(m_ext_data.raw_seqhdr_data) < param->ext_size)
                            return -1;

                        memset(&m_ext_data, 0, sizeof(m_ext_data));

                        m_ext_data.format.seqhdr_data_length = param->ext_size;
                        memcpy(m_ext_data.raw_seqhdr_data, param->ext_data, param->ext_size);

                        params.pExtVideoInfo = &m_ext_data;
                    }

                    cuCtxPushCurrent(m_gpu_ctx);
                    chk = m_funcs->cuvidCreateVideoParser(&m_parser, &params);
                    cuCtxPopCurrent(NULL);

                    if (chk != CUDA_SUCCESS)
                        return -1;

                    return 0;
                }

                virtual int update(abcdk_torch_frame_t **dst, const abcdk_torch_packet_t *src)
                {
                    CUVIDSOURCEDATAPACKET packet = {0};
                    CUresult cuda_chk;

                    if (!m_funcs)
                        return -1;

                    if (src)
                    {
                        packet.payload = (uint8_t *)src->data;
                        packet.payload_size = src->size;
                        packet.timestamp = (CUvideotimestamp)src->pts;
                        packet.flags |= CUVID_PKT_TIMESTAMP;

                        if (src->size == 0)
                            packet.flags |= CUVID_PKT_ENDOFSTREAM;

                        cuCtxPushCurrent(m_gpu_ctx);
                        cuda_chk = m_funcs->cuvidParseVideoData(m_parser, &packet);
                        cuCtxPopCurrent(NULL);

                        if (cuda_chk != CUDA_SUCCESS)
                            return -1;
                    }

                    if (dst)
                    {
                        abcdk_torch_frame_free(dst);

                        abcdk_queue_lock(m_frame_queue);
                        *dst = (abcdk_torch_frame_t *)abcdk_queue_pop(m_frame_queue);
                        abcdk_queue_unlock(m_frame_queue);

                        if (*dst)
                            return 1;
                    }

                    return 0;
                }
            };
        } // namespace vcodec
    } // namespace cuda
} // namespace abcdk


#endif // FFNV_CUDA_DYNLINK_LOADER_H
#endif // __cuda_cuda_h__

#endif // ABCDK_NVIDIA_VCODEC_DECODER_FFNV_HXX