/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_DECODER_FFNV_HXX
#define ABCDK_CUDA_VIDEO_DECODER_FFNV_HXX

#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"
#include "video_decoder.cu.hxx"
#include "video_util.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef AVCODEC_AVCODEC_H
#ifdef FFNV_CUDA_DYNLINK_LOADER_H

namespace abcdk
{
    namespace cuda
    {
        namespace video
        {
            class decoder_ffnv : public decoder
            {
            public:
                static decoder *create()
                {
                    decoder *ctx = new decoder_ffnv();
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

                static unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec, unsigned int nWidth, unsigned int nHeight)
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

                static int CUDAAPI HandleVideoSequenceProc(void *pUserData, CUVIDEOFORMAT *pVideoFormat)
                {
                    return ((decoder_ffnv *)pUserData)->VideoSequenceProc(pVideoFormat);
                }

                static int CUDAAPI HandlePictureDecodeProc(void *pUserData, CUVIDPICPARAMS *pPicParams)
                {
                    return ((decoder_ffnv *)pUserData)->PictureDecodeProc(pPicParams);
                }

                static int CUDAAPI HandlePictureDisplayProc(void *pUserData, CUVIDPARSERDISPINFO *pDispInfo)
                {
                    return ((decoder_ffnv *)pUserData)->PictureDisplayProc(pDispInfo);
                }

            private:
                CuvidFunctions *m_funcs;

                CUcontext m_gpu_ctx;
                CUvideoctxlock m_ctx_lock;

                CUVIDEOFORMATEX m_vidfmt_ext;

                CUvideoparser m_parser;
                CUvideodecoder m_decoder;

                CUVIDEOFORMAT m_videoformat;
                int m_nPicNumInDecodeOrder[32];
                int m_nDecodePicCnt;

            public:
                decoder_ffnv()
                {
                    m_funcs = NULL;
                    cuvid_load_functions(&m_funcs, NULL);

                    m_gpu_ctx = NULL;
                    m_ctx_lock = NULL;
                    m_parser = NULL;
                    m_decoder = NULL;

                    memset(&m_videoformat, 0, sizeof(CUVIDEOFORMAT));
                    memset(&m_nPicNumInDecodeOrder[0], 0, sizeof(int) * 32);

                    m_nDecodePicCnt = 0;
                }

                virtual ~decoder_ffnv()
                {
                    close();

                    cuvid_free_functions(&m_funcs);
                }

            protected:
                int VideoSequenceProc(CUVIDEOFORMAT *pVideoFormat)
                {
                    CUresult chk;

                    /*Copy*/
                    m_videoformat = *pVideoFormat;

                    CUVIDDECODECAPS decodecaps;
                    memset(&decodecaps, 0, sizeof(decodecaps));
                    decodecaps.eCodecType = m_videoformat.codec;
                    decodecaps.eChromaFormat = m_videoformat.chroma_format;
                    decodecaps.nBitDepthMinus8 = m_videoformat.bit_depth_luma_minus8;

                    cuCtxPushCurrent(m_gpu_ctx);
                    chk = m_funcs->cuvidGetDecoderCaps(&decodecaps);
                    cuCtxPopCurrent(NULL);

                    /*也许不支持*/
                    if ((chk != CUDA_SUCCESS) || !decodecaps.bIsSupported)
                        return 0;

                    int nDecodeSurface = GetNumDecodeSurfaces(m_videoformat.codec, m_videoformat.coded_width, m_videoformat.coded_height);

                    CUVIDDECODECREATEINFO vdecodecinfo = {0};
                    vdecodecinfo.CodecType = m_videoformat.codec;
                    vdecodecinfo.ChromaFormat = m_videoformat.chroma_format;
                    vdecodecinfo.OutputFormat = (m_videoformat.bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12);
                    vdecodecinfo.bitDepthMinus8 = m_videoformat.bit_depth_luma_minus8;
                    vdecodecinfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
                    vdecodecinfo.ulNumOutputSurfaces = nDecodeSurface; // 2;
                    vdecodecinfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
                    vdecodecinfo.ulNumDecodeSurfaces = nDecodeSurface;
                    vdecodecinfo.vidLock = m_ctx_lock;
                    vdecodecinfo.ulWidth = m_videoformat.coded_width;
                    vdecodecinfo.ulHeight = m_videoformat.coded_height;
                    vdecodecinfo.ulMaxWidth = m_videoformat.coded_width;
                    vdecodecinfo.ulMaxHeight = m_videoformat.coded_height;
                    vdecodecinfo.display_area.top = m_videoformat.display_area.top;
                    vdecodecinfo.display_area.bottom = m_videoformat.display_area.bottom;
                    vdecodecinfo.display_area.left = m_videoformat.display_area.left;
                    vdecodecinfo.display_area.right = m_videoformat.display_area.right;
                    vdecodecinfo.ulTargetWidth = abcdk_align(m_videoformat.display_area.right - m_videoformat.display_area.left, 2);
                    vdecodecinfo.ulTargetHeight = abcdk_align(m_videoformat.display_area.bottom - m_videoformat.display_area.top, 2);
                    vdecodecinfo.target_rect.top = 0;
                    vdecodecinfo.target_rect.bottom = vdecodecinfo.ulTargetHeight;
                    vdecodecinfo.target_rect.left = 0;
                    vdecodecinfo.target_rect.right = vdecodecinfo.ulTargetWidth;

                    cuCtxPushCurrent(m_gpu_ctx);
                    chk = m_funcs->cuvidCreateDecoder(&m_decoder, &vdecodecinfo);
                    cuCtxPopCurrent(NULL);

                    assert(chk == CUDA_SUCCESS);

                    return nDecodeSurface;
                }

                int PictureDecodeProc(CUVIDPICPARAMS *pPicParams)
                {
                    if (!m_decoder)
                        return 0;

                    m_nPicNumInDecodeOrder[pPicParams->CurrPicIdx] = m_nDecodePicCnt++;

                    CUresult chk = m_funcs->cuvidDecodePicture(m_decoder, pPicParams);
                    assert(chk == CUDA_SUCCESS);

                    return 1;
                }

                int PictureDisplayProc(CUVIDPARSERDISPINFO *pDispInfo)
                {
                    CUVIDPROCPARAMS params = {0};
                    CUdeviceptr dpSrcFrame = 0;
                    unsigned int dpSrcPitch = 0;
                    int width,height;
                    enum AVPixelFormat pixfmt;
                    uint8_t *src_data[4] = {0};
                    int src_linesize[4] = {0};
                    AVFrame *avframe_src;
                    CUresult chk;

                    if (!m_decoder)
                        return 0;
            
                    cuCtxPushCurrent(m_gpu_ctx);
                                
                    params.progressive_frame = pDispInfo->progressive_frame;
                    params.second_field = pDispInfo->repeat_first_field + 1;
                    params.top_field_first = pDispInfo->top_field_first;
                    params.unpaired_field = pDispInfo->repeat_first_field < 0;
                    params.output_stream = 0;
            
                    int64_t pts = pDispInfo->timestamp;
            
                    chk = m_funcs->cuvidMapVideoFrame(m_decoder, pDispInfo->picture_index, &dpSrcFrame, &dpSrcPitch, &params);
                    if (chk == CUDA_SUCCESS)
                    {
                        CUVIDGETDECODESTATUS DecodeStatus;
                        memset(&DecodeStatus,0,sizeof(DecodeStatus));
                        chk = m_funcs->cuvidGetDecodeStatus(m_decoder, pDispInfo->picture_index, &DecodeStatus);
                        if (chk == CUDA_SUCCESS && (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error || DecodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed))
                        {
                            abcdk_trace_printf(LOG_WARNING, "Decode Error occurred for picture %d", m_nPicNumInDecodeOrder[pDispInfo->picture_index]);
                        }
                    }

                    width = m_videoformat.display_area.right - m_videoformat.display_area.left;
                    height = m_videoformat.display_area.bottom - m_videoformat.display_area.top;
                    pixfmt = (m_videoformat.bit_depth_luma_minus8 ? AV_PIX_FMT_NV16 : AV_PIX_FMT_NV12);

                    src_linesize[0] = dpSrcPitch;
                    src_linesize[1] = dpSrcPitch;

                    abcdk_avimage_fill_pointers(src_data, src_linesize, height, pixfmt, (void **)dpSrcFrame);

                    avframe_src = abcdk_cuda_avframe_alloc(width,height,pixfmt,4);
                    if(avframe_src)
                    {
                        abcdk_cuda_avimage_copy(avframe_src->data,avframe_src->linesize,0,src_data,src_linesize,0,width,height,pixfmt);
                    }
                    else
                    {
                        abcdk_trace_printf(LOG_WARNING, "内存不足。");
                        return 1;
                    }
                    
                    chk = m_funcs->cuvidUnmapVideoFrame(m_decoder, dpSrcFrame);
                    assert(chk == CUDA_SUCCESS);
            
                    cuCtxPopCurrent(NULL);
                    
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

                    if (m_gpu_ctx)
                        cuCtxPopCurrent(NULL);

                    abcdk_cuda_destroy_ctx(&m_gpu_ctx);
                }

                virtual int open(abcdk_option_t *cfg)
                {
                    return -1;
                }

                virtual int sync(AVCodecContext *opt)
                {
                    return -1;
                }

                virtual int update(AVFrame **dst, const AVPacket *src)
                {
                    return -1;
                }
            };
        } // namespace video
    } // namespace cuda
} // namespace abcdk

#endif // FFNV_CUDA_DYNLINK_LOADER_H
#endif // AVCODEC_AVCODEC_H
#endif // AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_DECODER_FFNV_HXX