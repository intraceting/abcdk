/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"
#include "vdec.hxx"

#ifdef __x86_64__

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace vdec
        {
            typedef struct _metadata
            {
                CuvidFunctions *funcs;

                context::metadata_t *rt_ctx;

                abcdk_xpu_vcodec_params_t params;

                std::vector<uint8_t> ext_data;

                cudaVideoCodec nv_codec_id;

                CUvideoctxlock nv_ctx_lock;
                CUVIDEOFORMATEX nv_ext_data;
                CUvideoparser nv_parser;
                CUvideodecoder nv_decoder;
                CUVIDEOFORMAT nv_video_fmt;
                CUVIDDECODECREATEINFO nv_dec_info;
                size_t nv_pic_dec_order[32];
                size_t nv_pic_dec_count;

                std::mutex nv_frame_mutex;
                std::queue<image::metadata_t *> nv_frame_queue;

            } metadata_t;

            static unsigned long _get_dec_surfaces(cudaVideoCodec eCodec, unsigned int nWidth, unsigned int nHeight)
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

            static int CUDAAPI _sequence(metadata_t *ctx, CUVIDEOFORMAT *format)
            {
                int nv_dec_surface;
                CUresult nv_chk;

                ctx->nv_video_fmt = *format;

                CUVIDDECODECAPS nv_dec_caps = {.eCodecType = cudaVideoCodec_NumCodecs};

                nv_dec_caps.eCodecType = ctx->nv_video_fmt.codec;
                nv_dec_caps.eChromaFormat = ctx->nv_video_fmt.chroma_format;
                nv_dec_caps.nBitDepthMinus8 = ctx->nv_video_fmt.bit_depth_luma_minus8;
                
                nv_chk = ctx->funcs->cuvidGetDecoderCaps(&nv_dec_caps);
                if (nv_chk != CUDA_SUCCESS)
                    return 0;

                /*也许不支持.*/
                if (!nv_dec_caps.bIsSupported)
                    return 0;

                memset(&ctx->nv_dec_info,0,sizeof(ctx->nv_dec_info));
                
                ctx->nv_dec_info.ulWidth = ctx->nv_video_fmt.coded_width;
                ctx->nv_dec_info.ulHeight = ctx->nv_video_fmt.coded_height;
                
                nv_dec_surface = _get_dec_surfaces(ctx->nv_video_fmt.codec, ctx->nv_video_fmt.coded_width, ctx->nv_video_fmt.coded_height);
                ctx->nv_dec_info.ulNumDecodeSurfaces = nv_dec_surface;//内部缓存数量.

                ctx->nv_dec_info.CodecType = ctx->nv_video_fmt.codec;
                ctx->nv_dec_info.ChromaFormat = ctx->nv_video_fmt.chroma_format;

                ctx->nv_dec_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
                ctx->nv_dec_info.bitDepthMinus8 = nv_dec_caps.nBitDepthMinus8;

                ctx->nv_dec_info.ulMaxWidth = ctx->nv_video_fmt.coded_width;
                ctx->nv_dec_info.ulMaxHeight = ctx->nv_video_fmt.coded_height;

                ctx->nv_dec_info.display_area.top = ctx->nv_video_fmt.display_area.top;
                ctx->nv_dec_info.display_area.bottom = ctx->nv_video_fmt.display_area.bottom;
                ctx->nv_dec_info.display_area.left = ctx->nv_video_fmt.display_area.left;
                ctx->nv_dec_info.display_area.right = ctx->nv_video_fmt.display_area.right;

                ctx->nv_dec_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
                
                if (nv_dec_caps.eChromaFormat == cudaVideoChromaFormat_420 || nv_dec_caps.eChromaFormat == cudaVideoChromaFormat_Monochrome)
                    ctx->nv_dec_info.OutputFormat = ctx->nv_dec_info.bitDepthMinus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
                else if (nv_dec_caps.eChromaFormat == cudaVideoChromaFormat_444)
                    ctx->nv_dec_info.OutputFormat = ctx->nv_dec_info.bitDepthMinus8 ? cudaVideoSurfaceFormat_YUV444_16Bit : cudaVideoSurfaceFormat_YUV444;
                
                ctx->nv_dec_info.ulTargetWidth = abcdk_align(ctx->nv_video_fmt.display_area.right - ctx->nv_video_fmt.display_area.left, 2);
                ctx->nv_dec_info.ulTargetHeight = abcdk_align(ctx->nv_video_fmt.display_area.bottom - ctx->nv_video_fmt.display_area.top, 2);

                ctx->nv_dec_info.ulNumOutputSurfaces = 5; // map,map,map,map,map, unmap,unmap,unmap,unmap,unmap;
                ctx->nv_dec_info.vidLock = ctx->nv_ctx_lock;

                ctx->nv_dec_info.target_rect.top = 0;
                ctx->nv_dec_info.target_rect.bottom = ctx->nv_dec_info.ulTargetHeight;
                ctx->nv_dec_info.target_rect.left = 0;
                ctx->nv_dec_info.target_rect.right = ctx->nv_dec_info.ulTargetWidth;

                nv_chk = ctx->funcs->cuvidCreateDecoder(&ctx->nv_decoder, &ctx->nv_dec_info);
                if (nv_chk != CUDA_SUCCESS)
                    return 0;

                return nv_dec_surface;
            }

            static int CUDAAPI _sequence_cb(void *userdata, CUVIDEOFORMAT *format)
            {
                metadata_t *ctx = (metadata_t *)userdata;
                int chk;

                context::current_push(ctx->rt_ctx);
                chk = _sequence(ctx,format);
                context::current_pop(ctx->rt_ctx);

                return chk;
            }

            static int CUDAAPI _decode(metadata_t *ctx, CUVIDPICPARAMS *params)
            {
                CUresult nv_chk;

                ctx->nv_pic_dec_order[params->CurrPicIdx] = ctx->nv_pic_dec_count++;

                nv_chk = ctx->funcs->cuvidDecodePicture(ctx->nv_decoder, params);
                if (nv_chk != CUDA_SUCCESS)
                    return 0;

                return 1;
            }

            static int CUDAAPI _decode_cb(void *userdata, CUVIDPICPARAMS *params)
            {
                metadata_t *ctx = (metadata_t *)userdata;
                int chk;

                context::current_push(ctx->rt_ctx);
                chk = _decode(ctx, params);
                context::current_pop(ctx->rt_ctx);

                return chk;
            }

            static int CUDAAPI _display(metadata_t *ctx, CUVIDPARSERDISPINFO *info)
            {
                CUdeviceptr nv_frame = 0;
                unsigned int nv_pitch = 0;
                CUresult nv_chk;

                CUVIDPROCPARAMS nv_params = {0};
                CUVIDGETDECODESTATUS nv_status = {.decodeStatus = cuvidDecodeStatus_Invalid};

                nv_params.progressive_frame = info->progressive_frame;
                nv_params.second_field = info->repeat_first_field + 1;
                nv_params.top_field_first = info->top_field_first;
                nv_params.unpaired_field = info->repeat_first_field < 0;
                nv_params.output_stream = 0;

                nv_chk = ctx->funcs->cuvidMapVideoFrame(ctx->nv_decoder, info->picture_index, &nv_frame, &nv_pitch, &nv_params);
                if (nv_chk != CUDA_SUCCESS)
                    return 0;

                nv_chk = ctx->funcs->cuvidGetDecodeStatus(ctx->nv_decoder, info->picture_index, &nv_status);
                if (nv_chk != CUDA_SUCCESS)
                    return 0;

                if (nv_status.decodeStatus == cuvidDecodeStatus_Error || nv_status.decodeStatus == cuvidDecodeStatus_Error_Concealed)
                {
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("Decode Error occurred for picture %d"), ctx->nv_pic_dec_order[info->picture_index]);
                }

                int width = ctx->nv_video_fmt.display_area.right - ctx->nv_video_fmt.display_area.left;
                int height = ctx->nv_video_fmt.display_area.bottom - ctx->nv_video_fmt.display_area.top;
                abcdk_xpu_pixfmt_t pixfmt = ABCDK_XPU_PIXFMT_NONE;

                if (ctx->nv_dec_info.OutputFormat == cudaVideoSurfaceFormat_NV12)
                    pixfmt = ABCDK_XPU_PIXFMT_NV12;
                else if (ctx->nv_dec_info.OutputFormat == cudaVideoSurfaceFormat_P016)
                    pixfmt = ABCDK_XPU_PIXFMT_P016;
                else if (ctx->nv_dec_info.OutputFormat == cudaVideoSurfaceFormat_YUV444)
                    pixfmt = ABCDK_XPU_PIXFMT_YUV444P;
                else if (ctx->nv_dec_info.OutputFormat == cudaVideoSurfaceFormat_YUV444_16Bit)
                    pixfmt = ABCDK_XPU_PIXFMT_YUV444P16;
                else
                    ABCDK_TRACE_ASSERT(0,ABCDK_GETTEXT("占位, 代码不可能走到这里."));

                image::metadata_t *tmp_src = image::alloc();
                if (!tmp_src)
                    return 0;

                tmp_src->pts = info->timestamp;

                tmp_src->width = width;
                tmp_src->height = height;
                tmp_src->format = pixfmt::local_to_ffmpeg(pixfmt);

                tmp_src->linesize[0] = nv_pitch;
                tmp_src->linesize[1] = nv_pitch;
                tmp_src->linesize[2] = tmp_src->linesize[3] = 0;
                
                abcdk_ffmpeg_image_fill_pointer(tmp_src->data, tmp_src->linesize, tmp_src->height, (AVPixelFormat)tmp_src->format, (void *)nv_frame);

                ctx->nv_frame_mutex.lock();
                ctx->nv_frame_queue.push(tmp_src);
                ctx->nv_frame_mutex.unlock();

                return 1;
            }

            static int CUDAAPI _display_cb(void *userdata, CUVIDPARSERDISPINFO *info)
            {
                metadata_t *ctx = (metadata_t *)userdata;
                int chk;

                context::current_push(ctx->rt_ctx);
                chk = _display(ctx, info);
                context::current_pop(ctx->rt_ctx);

                return chk;
            }

            static int _init(metadata_t *ctx)
            {
                CUresult nv_chk;

                nv_chk = ctx->funcs->cuvidCtxLockCreate(&ctx->nv_ctx_lock, ctx->rt_ctx->cu_ctx);
                if (nv_chk != CUDA_SUCCESS)
                    return -EPERM;

                CUVIDPARSERPARAMS nv_parse_params = {.CodecType = ctx->nv_codec_id};
                nv_parse_params.ulMaxNumDecodeSurfaces = 25;
                nv_parse_params.ulMaxDisplayDelay = 4;
                nv_parse_params.pUserData = ctx;
                nv_parse_params.pfnSequenceCallback = _sequence_cb;
                nv_parse_params.pfnDecodePicture = _decode_cb;
                nv_parse_params.pfnDisplayPicture = _display_cb;
                nv_parse_params.pExtVideoInfo = NULL;

                if (ctx->params.ext_data != NULL && ctx->params.ext_size > 0)
                {
                    /*空间有限.*/
                    if (sizeof(ctx->nv_ext_data.raw_seqhdr_data) < ctx->params.ext_size)
                        return -EINVAL;

                    memset(&ctx->nv_ext_data, 0, sizeof(ctx->nv_ext_data));

                    ctx->nv_ext_data.format.seqhdr_data_length = ctx->params.ext_size;
                    memcpy(ctx->nv_ext_data.raw_seqhdr_data, ctx->params.ext_data, ctx->params.ext_size);

                    nv_parse_params.pExtVideoInfo = &ctx->nv_ext_data;
                }

                nv_chk = ctx->funcs->cuvidCreateVideoParser(&ctx->nv_parser, &nv_parse_params);
                if (nv_chk != CUDA_SUCCESS)
                    return -EPERM;

                return 0;
            }

            static int _send_packet(metadata_t *ctx, const void *src_data, size_t src_size, int64_t ts)
            {
                CUVIDSOURCEDATAPACKET nv_packet = {0};
                CUresult nv_chk;

                if (src_data != NULL && src_size > 0)
                {
                    nv_packet.payload = (uint8_t*)src_data;
                    nv_packet.payload_size = src_size;
                    nv_packet.timestamp = (CUvideotimestamp)ts;
                    nv_packet.flags |= CUVID_PKT_TIMESTAMP;
                }
                else
                {
                    nv_packet.flags |= CUVID_PKT_ENDOFSTREAM;
                }

                nv_chk = ctx->funcs->cuvidParseVideoData(ctx->nv_parser, &nv_packet);
                if (nv_chk != CUDA_SUCCESS)
                    return -EINVAL;

                return 1;
            }

            static int _recv_frame(metadata_t *ctx, image::metadata_t **dst, int64_t *ts)
            {
                CUresult nv_chk;
                int chk;

                // 以下代码全部加锁.
                std::lock_guard<std::mutex> auto_lock(ctx->nv_frame_mutex);

                if (ctx->nv_frame_queue.size() <= 0)
                    return 0;

                image::metadata_t *one = ctx->nv_frame_queue.front();
                ctx->nv_frame_queue.pop();

                chk = image::reset(dst, one->width, one->height, pixfmt::ffmpeg_to_local(one->format), 16, 0);
                if (chk == 0)
                {
                    image::copy(one, 0, *dst, 0);
                    *ts = one->pts;
                }

                nv_chk = ctx->funcs->cuvidUnmapVideoFrame(ctx->nv_decoder, (CUdeviceptr)one->data[0]); // Don't forget.
                assert(nv_chk == CUDA_SUCCESS);

                image::free(&one); // Don't forget.

                return (chk != 0 ? -1 : 1);
            }

            static void _clear_frame_queue(metadata_t *ctx)
            {
                CUresult nv_chk;

                ctx->nv_frame_mutex.lock();
                while (ctx->nv_frame_queue.size() > 0)
                {
                    image::metadata_t *one = ctx->nv_frame_queue.front();
                    ctx->nv_frame_queue.pop();

                    nv_chk = ctx->funcs->cuvidUnmapVideoFrame(ctx->nv_decoder, (CUdeviceptr)one->data[0]); // Don't forget.
                    assert(nv_chk == CUDA_SUCCESS);

                    image::free(&one); // Don't forget.
                }
                ctx->nv_frame_mutex.unlock();
            }

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                //必须先清理.
                _clear_frame_queue(ctx_p);

                if (ctx_p->nv_parser)
                    ctx_p->funcs->cuvidDestroyVideoParser(ctx_p->nv_parser);

                if (ctx_p->nv_decoder)
                    ctx_p->funcs->cuvidDestroyDecoder(ctx_p->nv_decoder);

                if (ctx_p->nv_ctx_lock)
                    ctx_p->funcs->cuvidCtxLockDestroy(ctx_p->nv_ctx_lock);

                context::unref(&ctx_p->rt_ctx);

                cuvid_free_functions(&ctx_p->funcs);

                delete ctx_p;
            }

            metadata_t *alloc()
            {
                metadata_t *ctx;

                ctx = new metadata_t;
                if (!ctx)
                    return NULL;

                ctx->funcs = NULL;
                ctx->rt_ctx = NULL;
                ctx->nv_ctx_lock = NULL;
                ctx->nv_parser = NULL;
                ctx->nv_decoder = NULL;
                ctx->nv_pic_dec_count = 0;

                cuvid_load_functions(&ctx->funcs, NULL);

                return ctx;
            }

            int setup(metadata_t *ctx, const abcdk_xpu_vcodec_params_t *params, context::metadata_t *rt_ctx)
            {
                CUresult nv_chk;
                int chk;

                ctx->rt_ctx = context::refer(rt_ctx);
                ctx->params = *params;

                ctx->ext_data.resize(params->ext_size);
                memcpy(ctx->ext_data.data(), params->ext_data, params->ext_size);

                //复制指针和长度.
                ctx->params.ext_data = ctx->ext_data.data();
                ctx->params.ext_size = ctx->ext_data.size();

                ctx->nv_codec_id = util::local_to_nvcodec((abcdk_xpu_vcodec_id_t)ctx->params.format);

                if (ctx->nv_codec_id <= cudaVideoCodec_MPEG1 || ctx->nv_codec_id >= cudaVideoCodec_NumCodecs)
                    return -EINVAL;

                chk = _init(ctx);
                if (chk != 0)
                    return chk;

                return 0;
            }

            int send_packet(metadata_t *ctx, const void *src_data, size_t src_size, int64_t ts)
            {
                return _send_packet(ctx, src_data, src_size, ts);
            }

            int recv_frame(metadata_t *ctx, image::metadata_t **dst, int64_t *ts)
            {
                return _recv_frame(ctx,dst,ts);
            }

        } // namespace vdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __x86_64__