/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_VCODEC_ENCODER_FFNV_HXX
#define ABCDK_NVIDIA_VCODEC_ENCODER_FFNV_HXX

#include "abcdk/torch/vcodec.h"
#include "abcdk/nvidia/nvidia.h"
#include "abcdk/nvidia/context.h"
#include "abcdk/nvidia/image.h"
#include "vcodec_encoder.cu.hxx"
#include "vcodec_util.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef FFNV_CUDA_DYNLINK_LOADER_H


namespace abcdk
{
    namespace cuda
    {
        namespace vcodec
        {
            class encoder_ffnv : public encoder
            {
            public:
                static encoder *create(CUcontext cuda_ctx)
                {
                    encoder *ctx = new encoder_ffnv(cuda_ctx);
                    if (!ctx)
                        return NULL;

                    return ctx;
                }

                static void destory(encoder **ctx)
                {
                    encoder *ctx_p;

                    if (!ctx || !*ctx)
                        return;

                    ctx_p = *ctx;
                    *ctx = NULL;

                    delete (encoder_ffnv *)ctx_p;
                }

            private:
                NvencFunctions *m_funcs;
                CUcontext m_gpu_ctx;

                std::vector<uint8_t> m_ext_data;
                NV_ENCODE_API_FUNCTION_LIST m_nvenc;
                void *m_encoder;
                NV_ENC_INITIALIZE_PARAMS m_params;
                NV_ENC_CONFIG m_config;

                int m_nExtraOutputDelay;
                NV_ENC_BUFFER_FORMAT m_bufmt;
                int m_nEncoderBuffer;
                int m_nOutputDelay;
                int32_t m_iToSend;
                int32_t m_iGot;

                std::vector<NV_ENC_OUTPUT_PTR> m_vBitstreamOutputBuffer;

                std::vector<abcdk_torch_image_t *> m_vInputFrames;
                std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResources;

                std::vector<NV_ENC_INPUT_PTR> m_vMappedInputBuffers;
                std::vector<NV_ENC_INPUT_PTR> m_vMappedRefBuffers;

                std::vector<void *> m_vpCompletionEvent;

            public:
                encoder_ffnv(CUcontext cuda_ctx)
                {
                    m_funcs = NULL;
                    nvenc_load_functions(&m_funcs, NULL);

                    m_gpu_ctx = cuda_ctx;

                    m_nvenc = {NV_ENCODE_API_FUNCTION_LIST_VER};
                    m_encoder = NULL;

                    memset(&m_params, 0, sizeof(m_params));
                    memset(&m_config, 0, sizeof(m_config));

                    m_nExtraOutputDelay = 0;
                    m_nEncoderBuffer = 0;

                }

                virtual ~encoder_ffnv()
                {
                    close();

                    nvenc_free_functions(&m_funcs);
                }

            protected:
                void CreateDefaultEncoderParams(NV_ENC_INITIALIZE_PARAMS *pIntializeParams,
                                                int fps, int width, int height, cudaVideoCodec codec, NV_ENC_BUFFER_FORMAT bufmt)
                {

                    assert(fps > 0 && width > 0 && height > 0);
                    assert(cudaVideoCodec_HEVC == codec || cudaVideoCodec_H264 == codec);

                    if (!m_encoder || !pIntializeParams || !pIntializeParams->encodeConfig)
                        return;

                    GUID presetGuid = NV_ENC_PRESET_DEFAULT_GUID;
                    GUID codecGuid = NV_ENC_CODEC_HEVC_GUID;
                    if (cudaVideoCodec_H264 == codec)
                        codecGuid = NV_ENC_CODEC_H264_GUID;

                    memset(pIntializeParams->encodeConfig, 0, sizeof(NV_ENC_CONFIG));
                    auto pEncodeConfig = pIntializeParams->encodeConfig;
                    memset(pIntializeParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
                    pIntializeParams->encodeConfig = pEncodeConfig;

                    pIntializeParams->encodeConfig->version = NV_ENC_CONFIG_VER;
                    pIntializeParams->version = NV_ENC_INITIALIZE_PARAMS_VER;

                    pIntializeParams->encodeGUID = codecGuid;
                    pIntializeParams->presetGUID = presetGuid;
                    pIntializeParams->encodeWidth = width;
                    pIntializeParams->encodeHeight = height;
                    pIntializeParams->darWidth = width;
                    pIntializeParams->darHeight = height;
                    pIntializeParams->frameRateNum = fps;
                    pIntializeParams->frameRateDen = 1;
                    pIntializeParams->enablePTD = 1;
                    pIntializeParams->reportSliceOffsets = 0;
                    pIntializeParams->enableSubFrameWrite = 0;
                    pIntializeParams->maxEncodeWidth = width;
                    pIntializeParams->maxEncodeHeight = height;
                    pIntializeParams->enableMEOnlyMode = false;
#if defined(_WIN32)
                    pIntializeParams->enableEncodeAsync = true;
#endif

                    NV_ENC_PRESET_CONFIG presetConfig = {NV_ENC_PRESET_CONFIG_VER, {NV_ENC_CONFIG_VER}};
                    m_nvenc.nvEncGetEncodePresetConfig(m_encoder, codecGuid, presetGuid, &presetConfig);

                    memcpy(pIntializeParams->encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));

                    // pIntializeParams->encodeConfig->gopLength = NVENC_INFINITE_GOPLENGTH;
                    pIntializeParams->encodeConfig->gopLength = fps;    /*Constant I-Frames.*/
                    pIntializeParams->encodeConfig->frameIntervalP = 1; /*1: IPP*/
                    pIntializeParams->encodeConfig->frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;

#if 0
                    pIntializeParams->encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
                    if (pIntializeParams->presetGUID != NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID && pIntializeParams->presetGUID != NV_ENC_PRESET_LOSSLESS_HP_GUID)
                    {
                        pIntializeParams->encodeConfig->rcParams.constQP = {28, 31, 25};
                    }
#else
                    pIntializeParams->encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
                    pIntializeParams->encodeConfig->rcParams.averageBitRate = 15000 * 1000; // 平均比特率 15 Mbps
                    pIntializeParams->encodeConfig->rcParams.maxBitRate = 15000 * 1000;     // 最大比特率，通常等于平均比特率 15 Mbps
                    pIntializeParams->encodeConfig->rcParams.vbvBufferSize = 15000 * 1000;  // 缓冲区大小 15 Mbits
                    pIntializeParams->encodeConfig->rcParams.vbvInitialDelay = 500 * 1000;  // 初始延迟 500 Kbits
#endif

                    if (pIntializeParams->encodeGUID == NV_ENC_CODEC_H264_GUID)
                    {
                        if (bufmt == NV_ENC_BUFFER_FORMAT_YUV444 || bufmt == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
                        {
                            pIntializeParams->encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC = 3;
                        }

                        pIntializeParams->encodeConfig->encodeCodecConfig.h264Config.idrPeriod = pIntializeParams->encodeConfig->gopLength;
                        pIntializeParams->encodeConfig->encodeCodecConfig.h264Config.maxNumRefFrames = 1; // No B-Frames.
                    }
                    else if (pIntializeParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID)
                    {
                        pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 =
                            (bufmt == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || bufmt == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) ? 2 : 0;

                        if (bufmt == NV_ENC_BUFFER_FORMAT_YUV444 || bufmt == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
                        {
                            pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
                        }

                        pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = pIntializeParams->encodeConfig->gopLength;
                        pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = 1; // No B-Frames.
                    }
                }

                void GetSequenceParams(std::vector<uint8_t> &seqParams)
                {
                    if (!m_encoder)
                        return;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    uint8_t spsppsData[1024]; // Assume maximum spspps data is 1KB or less
                    memset(spsppsData, 0, sizeof(spsppsData));
                    NV_ENC_SEQUENCE_PARAM_PAYLOAD payload = {NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER};
                    uint32_t spsppsSize = 0;

                    payload.spsppsBuffer = spsppsData;
                    payload.inBufferSize = sizeof(spsppsData);
                    payload.outSPSPPSPayloadSize = &spsppsSize;

                    NVENCSTATUS chk = m_nvenc.nvEncGetSequenceParams(m_encoder, &payload);

                    if (NV_ENC_SUCCESS == chk)
                    {
                        seqParams.clear();
                        seqParams.insert(seqParams.end(), &spsppsData[0], &spsppsData[spsppsSize]);
                    }
                }

                void InitializeBitstreamBuffer()
                {
                    if (!m_encoder)
                        return;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    for (int i = 0; i < m_nEncoderBuffer; i++)
                    {
                        NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = {NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
                        NVENCSTATUS chk = m_nvenc.nvEncCreateBitstreamBuffer(m_encoder, &createBitstreamBuffer);
                        if (chk == NV_ENC_SUCCESS)
                            m_vBitstreamOutputBuffer.push_back(createBitstreamBuffer.bitstreamBuffer);
                        else
                            break;
                    }
                }

                void DestroyBitstreamBuffer()
                {
                    if (!m_encoder)
                        return;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    for (uint32_t i = 0; i < m_vBitstreamOutputBuffer.size(); i++)
                    {
                        if (m_vBitstreamOutputBuffer[i])
                        {
                            m_nvenc.nvEncDestroyBitstreamBuffer(m_encoder, m_vBitstreamOutputBuffer[i]);
                        }
                    }

                    m_vBitstreamOutputBuffer.clear();
                }

                void RegisterResources()
                {
                    if (!m_encoder)
                        return;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    m_vInputFrames.resize(m_nEncoderBuffer);
                    for (int i = 0; i < m_vInputFrames.size(); i++)
                    {
                        m_vInputFrames[i] = abcdk_cuda_image_create(m_params.encodeWidth, m_params.encodeHeight, ABCDK_TORCH_PIXFMT_RGB32, 1);

                        NV_ENC_REGISTER_RESOURCE registerResource = {NV_ENC_REGISTER_RESOURCE_VER};
                        registerResource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
                        registerResource.resourceToRegister = (void *)m_vInputFrames[i]->data[0];
                        registerResource.width = m_vInputFrames[i]->width;
                        registerResource.height = m_vInputFrames[i]->height;
                        registerResource.pitch = m_vInputFrames[i]->stride[0];
                        registerResource.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
                        NVENCSTATUS chk = m_nvenc.nvEncRegisterResource(m_encoder, &registerResource);
                        if (chk == NV_ENC_SUCCESS)
                            m_vRegisteredResources.push_back(registerResource.registeredResource);
                    }
                }

                void DestroyResources()
                {
                    if (!m_encoder)
                        return;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    /*数组内的成员是指针对象，必须逐个释放。*/
                    for(auto &t: m_vRegisteredResources)
                        m_nvenc.nvEncUnregisterResource(m_encoder,t);

                    m_vRegisteredResources.clear();

                    /*数组内的成员是指针对象，必须逐个释放。*/
                    for (auto &t : m_vInputFrames)
                        abcdk_torch_image_free(&t);

                    m_vInputFrames.clear();
                }

                void DoEncode(NV_ENC_INPUT_PTR inputBuffer, std::vector<std::vector<uint8_t>> &vPacket)
                {
                    if (!m_encoder)
                        return;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    int i = m_iToSend % m_nEncoderBuffer;

                    NV_ENC_PIC_PARAMS picParams = {NV_ENC_PIC_PARAMS_VER};
                    picParams.version = NV_ENC_PIC_PARAMS_VER;
                    picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
                    picParams.inputBuffer = inputBuffer;
                    picParams.bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR;
                    picParams.inputWidth = m_params.encodeWidth;
                    picParams.inputHeight = m_params.encodeHeight;
                    picParams.outputBitstream = m_vBitstreamOutputBuffer[i];
                    picParams.completionEvent = m_vpCompletionEvent[i];
                    NVENCSTATUS chk = m_nvenc.nvEncEncodePicture(m_encoder, &picParams);

                    if (chk == NV_ENC_SUCCESS || chk == NV_ENC_ERR_NEED_MORE_INPUT)
                    {
                        m_iToSend++;
                        GetEncodedPacket(m_vBitstreamOutputBuffer, vPacket, false);
                    }
                }

                void EndEncode(std::vector<std::vector<uint8_t>> &vPacket)
                {
                    if (!m_encoder)
                        return;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    NV_ENC_PIC_PARAMS picParams = {NV_ENC_PIC_PARAMS_VER};
                    picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
                    picParams.completionEvent = m_vpCompletionEvent[m_iToSend % m_nEncoderBuffer];

                    NVENCSTATUS chk = m_nvenc.nvEncEncodePicture(m_encoder, &picParams);

                    if (chk == NV_ENC_SUCCESS)
                        GetEncodedPacket(m_vBitstreamOutputBuffer, vPacket, false);
                }

                void GetEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR> &vOutputBuffer, std::vector<std::vector<uint8_t>> &vPacket, bool bOutputDelay)
                {
                    if (!m_encoder)
                        return;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    unsigned i = 0;
                    int iEnd = bOutputDelay ? m_iToSend - m_nOutputDelay : m_iToSend;
                    for (; m_iGot < iEnd; m_iGot++)
                    {

                        NV_ENC_LOCK_BITSTREAM lockBitstreamData = {NV_ENC_LOCK_BITSTREAM_VER};
                        lockBitstreamData.outputBitstream = vOutputBuffer[m_iGot % m_nEncoderBuffer];
                        lockBitstreamData.doNotWait = false;
                        m_nvenc.nvEncLockBitstream(m_encoder, &lockBitstreamData);

                        uint8_t *pData = (uint8_t *)lockBitstreamData.bitstreamBufferPtr;
                        if (vPacket.size() < i + 1)
                        {
                            vPacket.push_back(std::vector<uint8_t>());
                        }
                        vPacket[i].clear();
                        vPacket[i].insert(vPacket[i].end(), &pData[0], &pData[lockBitstreamData.bitstreamSizeInBytes]);
                        i++;

                        m_nvenc.nvEncUnlockBitstream(m_encoder, lockBitstreamData.outputBitstream);

                        if (m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer])
                        {
                            m_nvenc.nvEncUnmapInputResource(m_encoder, m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer]);
                            m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer] = nullptr;
                        }
                    }
                }

                int encode(const abcdk_torch_image_t *img, std::vector<std::vector<uint8_t>> &out)
                {
                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    if (img)
                    {
                        int i = m_iToSend % m_nEncoderBuffer;

                        abcdk_cuda_image_copy(m_vInputFrames[i], img);

                        NV_ENC_MAP_INPUT_RESOURCE mapInputResource = {NV_ENC_MAP_INPUT_RESOURCE_VER};
                        mapInputResource.registeredResource = m_vRegisteredResources[i];

                        NVENCSTATUS nvenc_chk = m_nvenc.nvEncMapInputResource(m_encoder, &mapInputResource);
                        if (NV_ENC_SUCCESS != nvenc_chk)
                            return -1;

                        m_vMappedInputBuffers[i] = mapInputResource.mappedResource;

                        DoEncode(m_vMappedInputBuffers[i], out);
                    }
                    else
                    {
                        EndEncode(out);
                    }

                    if (out.size() <= 0)
                        return 0;

                    return 1;
                }

            public:
                virtual void close()
                {
                    if (!m_funcs)
                        return;

                    if (m_gpu_ctx)
                        cuCtxPushCurrent(m_gpu_ctx);

                    DestroyBitstreamBuffer();
                    DestroyResources();

                    memset(&m_params, 0, sizeof(m_params));
                    memset(&m_config, 0, sizeof(m_config));
                    m_nExtraOutputDelay = 0;
                    m_nEncoderBuffer = 0;
                    m_nOutputDelay = 0;
                    m_iToSend = 0;
                    m_iGot = 0;
                    m_vpCompletionEvent.clear();
                    m_vMappedRefBuffers.clear();
                    m_vMappedInputBuffers.clear();
                    m_vRegisteredResources.clear();


                    if (m_encoder)
                        m_nvenc.nvEncDestroyEncoder(m_encoder);
                    m_encoder = NULL;

                    if (m_gpu_ctx)
                        cuCtxPopCurrent(NULL);
                }

                virtual int open(abcdk_torch_vcodec_param_t *param)
                {
                    uint32_t version = 0;
                    uint32_t currentVersion = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
                    int fps, width, height;
                    cudaVideoCodec nv_codecid;
                    
                    NVENCSTATUS chk;

                    assert(param != NULL);

                    if (!m_funcs)
                        return -1;

                    fps = param->fps_n/param->fps_d;
                    width = param->width;
                    height = param->height;
                    nv_codecid = (cudaVideoCodec)vcodec_to_nvcodec(param->format);

                    if (fps > 1000 || fps <= 0)
                        return -1;

                    if (width <= 0 || height <= 0)
                        return -1;

                    if (cudaVideoCodec_HEVC != nv_codecid && cudaVideoCodec_H264 != nv_codecid)
                        return -1;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    chk = m_funcs->NvEncodeAPIGetMaxSupportedVersion(&version);

                    if (NV_ENC_SUCCESS != chk)
                        return -1;

                    if (currentVersion <= version)
                    {
                        m_nvenc = {NV_ENCODE_API_FUNCTION_LIST_VER};
                        chk = m_funcs->NvEncodeAPICreateInstance(&m_nvenc);
                    }

                    if (NV_ENC_SUCCESS != chk)
                        return -1;

                    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = {NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
                    encodeSessionExParams.device = m_gpu_ctx;
                    encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
                    encodeSessionExParams.apiVersion = NVENCAPI_VERSION;

                    chk = m_nvenc.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &m_encoder);

                    if (NV_ENC_SUCCESS != chk)
                        return -1;

                    memset(&m_params, 0, sizeof(m_params));
                    m_params.version = NV_ENC_INITIALIZE_PARAMS_VER;

                    memset(&m_config, 0, sizeof(m_config));
                    m_config.version = NV_ENC_CONFIG_VER;

                    m_params.encodeConfig = &m_config;

                    CreateDefaultEncoderParams(&m_params, fps, width, height, nv_codecid, NV_ENC_BUFFER_FORMAT_ABGR);

                    chk = m_nvenc.nvEncInitializeEncoder(m_encoder, &m_params);

                    if (NV_ENC_SUCCESS != chk)
                        return -1;

                    m_nExtraOutputDelay = 1;
                    m_nEncoderBuffer = m_config.frameIntervalP + m_config.rcParams.lookaheadDepth + m_nExtraOutputDelay;
                    m_nOutputDelay = m_nEncoderBuffer - 1;
                    m_vMappedInputBuffers.resize(m_nEncoderBuffer, nullptr);
                    m_vpCompletionEvent.resize(m_nEncoderBuffer, nullptr);
                    m_iToSend = 0;
                    m_iGot = 0;

                    InitializeBitstreamBuffer();

                    RegisterResources();

                    if (m_vBitstreamOutputBuffer.size() == m_nEncoderBuffer &&
                        m_vRegisteredResources.size() == m_nEncoderBuffer &&
                        m_vInputFrames.size() == m_nEncoderBuffer)
                    {
                        GetSequenceParams(m_ext_data);
                    }
                    else
                    {
                        return -1;
                    }

                    /*输出扩展数据帧。*/
                    if (m_ext_data.size() > 0)
                    {
                        param->ext_data = m_ext_data.data();
                        param->ext_size = m_ext_data.size();
                    }

                    return 0;
                }

                virtual int update(abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src)
                {
                    abcdk_torch_frame_t *tmp_src = NULL;
                    std::vector<std::vector<uint8_t>> out;
                    int dst_size = 0, dst_off = 0;
                    int chk;

                    assert(dst != NULL);

                    if (!m_funcs)
                        return -1;

                    if (!m_encoder)
                        return -1;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    if (src)
                    {
                        if (src->img->pixfmt != ABCDK_TORCH_PIXFMT_RGB32)
                        {
                            tmp_src = abcdk_cuda_frame_create(src->img->width, src->img->height, ABCDK_TORCH_PIXFMT_RGB32, 1);
                            if (!tmp_src)
                                return -1;
                            
                            /*转换格式。*/
                            chk = abcdk_cuda_image_convert(tmp_src->img, src->img); 

                            if (chk == 0)
                            {
                                /*复制其它参数。*/
                                tmp_src->dts = src->dts;
                                tmp_src->pts = src->pts;

                                chk = update(dst, tmp_src);
                            }

                            abcdk_torch_frame_free(&tmp_src);

                            return chk;
                        }

                        chk = encode(src->img, out);
                        if (chk < 0)
                            return -1;
                    }
                    else
                    {
                        chk = encode(NULL, out);
                        if (chk < 0)
                            return -1;
                    }

                    if (out.size() <= 0)
                        return 0;

                    for (int j = 0; j < out.size(); j++)
                        dst_size += out[j].size();

                    chk = abcdk_torch_packet_reset(dst,dst_size);
                    if(chk != 0)
                        return -1;

                    for (int j = 0; j < out.size(); j++)
                    {
                        memcpy(ABCDK_PTR2VPTR((*dst)->data, dst_off), out[j].data(), out[j].size());
                        dst_off += out[j].size();
                    }

                    return 1;
                }
            };
        } // namespace video
    } // namespace cuda
} // namespace abcdk


#endif // FFNV_CUDA_DYNLINK_LOADER_H
#endif // __cuda_cuda_h__

#endif // ABCDK_NVIDIA_VCODEC_ENCODER_FFNV_HXX