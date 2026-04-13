/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"
#include "venc.hxx"

#ifdef __x86_64__

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace venc
        {
            typedef struct _metadata
            {
                NvencFunctions *funcs;

                context::metadata_t *rt_ctx;

                abcdk_xpu_vcodec_params_t params;

                std::vector<uint8_t> ext_data;

                cudaVideoCodec nv_codec_id;

                NV_ENCODE_API_FUNCTION_LIST nv_apis;
                void *nv_encoder;
                NV_ENC_BUFFER_FORMAT nv_bufmt;
                NV_ENC_INITIALIZE_PARAMS nv_params;
                NV_ENC_CONFIG nv_config;

                size_t recv_index;
                size_t send_index;
                size_t nv_buf_num;

                std::vector<NV_ENC_OUTPUT_PTR> nv_bs_out_bufs;
                std::vector<image::metadata_t *> nv_img_in_bufs;
                std::vector<NV_ENC_REGISTERED_PTR> nv_img_in_regs;
                std::vector<NV_ENC_INPUT_PTR> nv_img_in_maps;
                std::vector<void *> nv_vp_com_evts;

            } metadata_t;

            static inline bool operator==(const GUID &guid1, const GUID &guid2)
            {
                return !memcmp(&guid1, &guid2, sizeof(GUID));
            }

            static inline bool operator!=(const GUID &guid1, const GUID &guid2)
            {
                return !(guid1 == guid2);
            }

            static inline int _init(metadata_t *ctx)
            {
                uint32_t api_version = 0;
                uint32_t sdk_version = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
                NVENCSTATUS nv_chk;

                nv_chk = ctx->funcs->NvEncodeAPIGetMaxSupportedVersion(&api_version);
                if (NV_ENC_SUCCESS != nv_chk)
                    return -EPERM;

                if (sdk_version > api_version)
                    return -EACCES;

                ctx->nv_apis = {.version = NV_ENCODE_API_FUNCTION_LIST_VER};
                nv_chk = ctx->funcs->NvEncodeAPICreateInstance(&ctx->nv_apis);
                if (NV_ENC_SUCCESS != nv_chk)
                    return -EPERM;

                NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS nv_ses_param = {.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
                nv_ses_param.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
                nv_ses_param.device = ctx->rt_ctx->cu_ctx;
                nv_ses_param.apiVersion = NVENCAPI_VERSION;

                nv_chk = ctx->nv_apis.nvEncOpenEncodeSessionEx(&nv_ses_param, &ctx->nv_encoder);
                if (NV_ENC_SUCCESS != nv_chk)
                    return -EPERM;

                GUID preset_guid = NV_ENC_PRESET_DEFAULT_GUID;
                GUID codec_guid = NV_ENC_CODEC_HEVC_GUID;
                if (cudaVideoCodec_H264 == ctx->nv_codec_id)
                    codec_guid = NV_ENC_CODEC_H264_GUID;

                ctx->nv_config = {.version = NV_ENC_CONFIG_VER};
                ctx->nv_params = {.version = NV_ENC_INITIALIZE_PARAMS_VER, .encodeConfig = &ctx->nv_config};

                ctx->nv_params.encodeGUID = codec_guid;
                ctx->nv_params.presetGUID = preset_guid;
                ctx->nv_params.encodeWidth = ctx->params.width;
                ctx->nv_params.encodeHeight = ctx->params.height;
                ctx->nv_params.darWidth = ctx->params.width;
                ctx->nv_params.darHeight = ctx->params.height;
                ctx->nv_params.frameRateNum = ctx->params.fps_n;
                ctx->nv_params.frameRateDen = ctx->params.fps_d;
                ctx->nv_params.enablePTD = 1;
                ctx->nv_params.reportSliceOffsets = 0;
                ctx->nv_params.enableSubFrameWrite = 0;
                ctx->nv_params.maxEncodeWidth = ctx->params.width;
                ctx->nv_params.maxEncodeHeight = ctx->params.height;
                ctx->nv_params.enableMEOnlyMode = false;
                ctx->nv_params.enableEncodeAsync = false;

                NV_ENC_PRESET_CONFIG nv_preset_config = {.version = NV_ENC_PRESET_CONFIG_VER, .presetCfg = {NV_ENC_CONFIG_VER}};
                nv_chk = ctx->nv_apis.nvEncGetEncodePresetConfig(ctx->nv_encoder, codec_guid, preset_guid, &nv_preset_config);
                if (nv_chk != NV_ENC_SUCCESS)
                    return -EPERM;

                memcpy(ctx->nv_params.encodeConfig, &nv_preset_config.presetCfg, sizeof(NV_ENC_CONFIG));

                // ctx->nv_params.encodeConfig->gopLength = NVENC_INFINITE_GOPLENGTH;
                ctx->nv_params.encodeConfig->gopLength = ctx->params.iframe_interval;

                if (ctx->params.max_b_frames > 1)
                    ctx->nv_params.encodeConfig->frameIntervalP = 3; /*3: IBBP*/
                else if (ctx->params.max_b_frames > 0)
                    ctx->nv_params.encodeConfig->frameIntervalP = 2; /*2: IBP*/
                else                                                 
                    ctx->nv_params.encodeConfig->frameIntervalP = 1; /*1: IPP*/

                ctx->nv_params.encodeConfig->frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;

                if (ctx->params.hw_preset_type == 0)
                {
                    ctx->nv_params.encodeConfig->rcParams.zeroReorderDelay = 0;
                    ctx->nv_params.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
                    ctx->nv_params.encodeConfig->rcParams.constQP = {ctx->params.qmax, ctx->params.qmin};
                }
                else if (ctx->params.hw_preset_type == 1)
                {
                    ctx->nv_params.encodeConfig->rcParams.zeroReorderDelay = 1;
                    ctx->nv_params.encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ;
                }
                else if (ctx->params.hw_preset_type == 2)
                {
                    ctx->nv_params.encodeConfig->rcParams.zeroReorderDelay = 1;
                    ctx->nv_params.encodeConfig->rcParams.rateControlMode = (ctx->params.mode_vbr ? NV_ENC_PARAMS_RC_VBR_HQ : NV_ENC_PARAMS_RC_CBR_HQ);
                }
                else if (ctx->params.hw_preset_type == 3)
                {
                    ctx->nv_params.encodeConfig->rcParams.zeroReorderDelay = 0;
                    ctx->nv_params.encodeConfig->rcParams.rateControlMode = (ctx->params.mode_vbr ? NV_ENC_PARAMS_RC_VBR : NV_ENC_PARAMS_RC_CBR);
                }

                ctx->nv_params.encodeConfig->rcParams.averageBitRate = ctx->params.bitrate;  // 平均比特率.
                ctx->nv_params.encodeConfig->rcParams.maxBitRate = ctx->params.max_bitrate; // 最大比特率.

                ctx->nv_bufmt = NV_ENC_BUFFER_FORMAT_ABGR;

                if (ctx->nv_params.encodeGUID == NV_ENC_CODEC_H264_GUID)
                {
                    if (ctx->nv_bufmt == NV_ENC_BUFFER_FORMAT_YUV444 || ctx->nv_bufmt == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
                    {
                        ctx->nv_params.encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 3;
                    }

                    ctx->nv_params.encodeConfig->encodeCodecConfig.h264Config.idrPeriod = ctx->params.idr_interval;
                    ctx->nv_params.encodeConfig->encodeCodecConfig.h264Config.maxNumRefFrames = ctx->params.refs;
                    ctx->nv_params.encodeConfig->encodeCodecConfig.h264Config.repeatSPSPPS = ctx->params.insert_spspps_idr;

                    ctx->nv_params.encodeConfig->encodeCodecConfig.h264Config.level = NV_ENC_LEVEL_H264_51;
                }
                else if (ctx->nv_params.encodeGUID == NV_ENC_CODEC_HEVC_GUID)
                {
                    ctx->nv_params.encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 =
                        (ctx->nv_bufmt == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || ctx->nv_bufmt == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 2 : 0;

                    if (ctx->nv_bufmt == NV_ENC_BUFFER_FORMAT_YUV444 || ctx->nv_bufmt == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
                    {
                        ctx->nv_params.encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
                    }

                    ctx->nv_params.encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = ctx->params.idr_interval;
                    ctx->nv_params.encodeConfig->encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = ctx->params.refs;
                    ctx->nv_params.encodeConfig->encodeCodecConfig.hevcConfig.repeatSPSPPS = ctx->params.insert_spspps_idr;

                    ctx->nv_params.encodeConfig->encodeCodecConfig.hevcConfig.level = NV_ENC_LEVEL_HEVC_51;
                    ctx->nv_params.encodeConfig->encodeCodecConfig.hevcConfig.tier = NV_ENC_TIER_HEVC_MAIN;

                }

                nv_chk = ctx->nv_apis.nvEncInitializeEncoder(ctx->nv_encoder, &ctx->nv_params);
                if (NV_ENC_SUCCESS != nv_chk)
                    return -EINVAL;

                // ctx->nv_buf_num = ctx->nv_config.frameIntervalP + ctx->nv_config.rcParams.lookaheadDepth + 1;
                ctx->nv_buf_num = 5;

                for (int i = 0; i < ctx->nv_buf_num; i++)
                {
                    NV_ENC_CREATE_BITSTREAM_BUFFER nv_create_bs_buf = {.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
                    nv_chk = ctx->nv_apis.nvEncCreateBitstreamBuffer(ctx->nv_encoder, &nv_create_bs_buf);
                    if (nv_chk != NV_ENC_SUCCESS)
                        return -ENOMEM;

                    ctx->nv_bs_out_bufs.push_back(nv_create_bs_buf.bitstreamBuffer); // 输出缓存, 循环使用.

                    image::metadata_t *nv_img_in = image::create(ctx->params.width, ctx->params.height, ABCDK_XPU_PIXFMT_RGB32, 16, 0); // ctx->nv_bufmt
                    if (!nv_img_in)
                        return -ENOMEM;

                    ctx->nv_img_in_bufs.push_back(nv_img_in); // 输入缓存, 循环使用.

                    NV_ENC_REGISTER_RESOURCE nv_reg_res = {.version = NV_ENC_REGISTER_RESOURCE_VER, .resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR};

                    nv_reg_res.resourceToRegister = (void *)nv_img_in->data[0];
                    nv_reg_res.width = nv_img_in->width;
                    nv_reg_res.height = nv_img_in->height;
                    nv_reg_res.pitch = nv_img_in->linesize[0];
                    nv_reg_res.bufferFormat = ctx->nv_bufmt;
                    nv_reg_res.bufferUsage = NV_ENC_INPUT_IMAGE;

                    nv_chk = ctx->nv_apis.nvEncRegisterResource(ctx->nv_encoder, &nv_reg_res); // 注册输入缓存, 循环使用.
                    if (nv_chk != NV_ENC_SUCCESS)
                        return -ENOMEM;

                    ctx->nv_img_in_regs.push_back(nv_reg_res.registeredResource);

                    ctx->nv_img_in_maps.push_back(NULL);
                    ctx->nv_vp_com_evts.push_back(NULL);
                }

                uint32_t ext_size = 0;
                uint8_t ext_data[1024] = {0}; // Assume maximum spspps data is 1KB or less

                NV_ENC_SEQUENCE_PARAM_PAYLOAD payload = {.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER};
                payload.spsppsBuffer = ext_data;
                payload.inBufferSize = sizeof(ext_data);
                payload.outSPSPPSPayloadSize = &ext_size;

                nv_chk = ctx->nv_apis.nvEncGetSequenceParams(ctx->nv_encoder, &payload); // 获取编码信息.
                if (nv_chk != NV_ENC_SUCCESS)
                    return -EPERM;

                ctx->ext_data.clear();
                ctx->ext_data.insert(ctx->ext_data.end(), &ext_data[0], &ext_data[ext_size]);

                //复制指针和长度.
                ctx->params.ext_data = ctx->ext_data.data();
                ctx->params.ext_size = ctx->ext_data.size();

                return 0;
            }

            static inline int _recv_packet(metadata_t *ctx, abcdk_object_t **dst, int64_t *ts)
            {
                NVENCSTATUS nv_chk;
                int chk;

                if (ctx->recv_index >= ctx->send_index)
                    return 0;

                int idx = ctx->recv_index % ctx->nv_buf_num;

                NV_ENC_LOCK_BITSTREAM nv_lock_bs = {.version = NV_ENC_LOCK_BITSTREAM_VER};
                nv_lock_bs.outputBitstream = ctx->nv_bs_out_bufs[idx];
                nv_lock_bs.doNotWait = false;

                nv_chk = ctx->nv_apis.nvEncLockBitstream(ctx->nv_encoder, &nv_lock_bs); // 加锁输出缓存.
                if (nv_chk != NV_ENC_SUCCESS)
                    return -EPERM;

                abcdk_object_unref(dst);
                *dst = abcdk_object_copyfrom(nv_lock_bs.bitstreamBufferPtr, nv_lock_bs.bitstreamSizeInBytes);

                *ts = nv_lock_bs.outputTimeStamp;

                nv_chk = ctx->nv_apis.nvEncUnlockBitstream(ctx->nv_encoder, nv_lock_bs.outputBitstream); // 解锁输出缓存.
                assert(nv_chk == NV_ENC_SUCCESS);

                if (ctx->nv_img_in_maps[idx])
                {
                    nv_chk = ctx->nv_apis.nvEncUnmapInputResource(ctx->nv_encoder, ctx->nv_img_in_maps[idx]); // 反映射输入缓存.
                    assert(nv_chk == NV_ENC_SUCCESS);

                    ctx->nv_img_in_maps[idx] = NULL;
                }

                ctx->recv_index += 1;
                return 1;
            }

            static inline int _send_frame(metadata_t *ctx, const image::metadata_t *src, int64_t ts)
            {
                NVENCSTATUS nv_chk;

                if (ctx->send_index - ctx->recv_index >= ctx->nv_buf_num)
                    return 0;

                int idx = ctx->send_index % ctx->nv_buf_num;

                NV_ENC_PIC_PARAMS nv_pic_params = {.version = NV_ENC_PIC_PARAMS_VER};

                if (src)
                {
                    image::copy(src, 0, ctx->nv_img_in_bufs[idx], 0); // 向输入缓存复制图像.

                    NV_ENC_MAP_INPUT_RESOURCE nv_map_in_res = {.version = NV_ENC_MAP_INPUT_RESOURCE_VER};
                    nv_map_in_res.registeredResource = ctx->nv_img_in_regs[idx];

                    nv_chk = ctx->nv_apis.nvEncMapInputResource(ctx->nv_encoder, &nv_map_in_res); // 映射输入缓存.
                    if (NV_ENC_SUCCESS != nv_chk)
                        return -EPERM;

                    ctx->nv_img_in_maps[idx] = nv_map_in_res.mappedResource;

                    nv_pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
                    nv_pic_params.inputBuffer = ctx->nv_img_in_maps[idx];
                    nv_pic_params.bufferFmt = ctx->nv_bufmt;
                    nv_pic_params.inputWidth = ctx->params.width;
                    nv_pic_params.inputHeight = ctx->params.height;
                    nv_pic_params.inputPitch = ctx->nv_img_in_bufs[idx]->linesize[0];

                    nv_pic_params.inputTimeStamp = (uint64_t)ts;
                    nv_pic_params.outputBitstream = ctx->nv_bs_out_bufs[idx];
                    nv_pic_params.completionEvent = ctx->nv_vp_com_evts[idx];

                    nv_chk = ctx->nv_apis.nvEncEncodePicture(ctx->nv_encoder, &nv_pic_params); // 编码图像.
                    if (nv_chk == NV_ENC_SUCCESS || nv_chk == NV_ENC_ERR_NEED_MORE_INPUT)
                    {
                        ctx->send_index += 1;
                        return 1;
                    }
                    else if (nv_chk == NV_ENC_ERR_ENCODER_BUSY)
                    {
                        return 0;
                    }
                }
                else
                {
                    nv_pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
                    nv_pic_params.completionEvent = ctx->nv_vp_com_evts[idx];

                    nv_chk = ctx->nv_apis.nvEncEncodePicture(ctx->nv_encoder, &nv_pic_params);
                    if (nv_chk == NV_ENC_SUCCESS)
                        return 1;
                }

                return -1;
            }

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                /*动态创建的对象, 必须逐个释放.*/
                for (auto &one : ctx_p->nv_bs_out_bufs)
                    ctx_p->nv_apis.nvEncDestroyBitstreamBuffer(ctx_p->nv_encoder, one);

                /*动态创建的对象, 必须逐个释放.*/
                for (auto &one : ctx_p->nv_img_in_regs)
                    ctx_p->nv_apis.nvEncUnregisterResource(ctx_p->nv_encoder, one);

                /*动态创建的对象, 必须逐个释放.*/
                for (auto &one : ctx_p->nv_img_in_bufs)
                    image::free(&one);

                if (ctx_p->nv_encoder)
                    ctx_p->nv_apis.nvEncDestroyEncoder(ctx_p->nv_encoder);

                context::unref(&ctx_p->rt_ctx);

                nvenc_free_functions(&ctx_p->funcs);

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
                ctx->nv_encoder = NULL;
                ctx->recv_index = 0;
                ctx->send_index = 0;

                nvenc_load_functions(&ctx->funcs, NULL);

                return ctx;
            }

            int setup(metadata_t *ctx, const abcdk_xpu_vcodec_params_t *params, context::metadata_t *rt_ctx)
            {

                int chk;

                ctx->rt_ctx = context::refer(rt_ctx);
                ctx->params = *params;
                ctx->params.ext_data = NULL;
                ctx->params.ext_size = 0;

                ctx->nv_codec_id = util::local_to_nvcodec((abcdk_xpu_vcodec_id_t)ctx->params.format);

                if (ctx->params.fps_n <= 0 || ctx->params.fps_d <= 0)
                    return -EINVAL;

                if (ctx->params.width <= 0 || ctx->params.height <= 0)
                    return -EINVAL;

                if (cudaVideoCodec_HEVC != ctx->nv_codec_id && cudaVideoCodec_H264 != ctx->nv_codec_id)
                    return -EINVAL;

                chk = _init(ctx);
                if (chk != 0)
                    return chk;

                return 0;
            }

            int get_params(metadata_t *ctx, abcdk_xpu_vcodec_params_t *params)
            {
                *params = ctx->params;
                
                return 0;
            }

            int recv_packet(metadata_t *ctx, abcdk_object_t **dst, int64_t *ts)
            {
                return _recv_packet(ctx, dst, ts);
            }

            int send_frame(metadata_t *ctx, const image::metadata_t *src, int64_t ts)
            {
                image::metadata_t *tmp_src;
                int chk;

                if (src != NULL && src->format != AV_PIX_FMT_RGB32)
                {
                    tmp_src = image::create(src->width, src->height, ABCDK_XPU_PIXFMT_RGB32, 16, 0);
                    if (!tmp_src)
                        return -ENOMEM;

                    chk = imgproc::convert(src, tmp_src);
                    if (chk != 0)
                    {
                        image::free(&tmp_src);
                        return -EPERM;
                    }

                    chk = send_frame(ctx, tmp_src, ts);
                    image::free(&tmp_src);

                    return chk;
                }

                return _send_frame(ctx, src, ts);
            }

        } // namespace venc
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __x86_64__