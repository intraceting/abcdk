/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_VCODEC_ENCODER_AARCH64_HXX
#define ABCDK_TORCH_NVIDIA_VCODEC_ENCODER_AARCH64_HXX

#include "abcdk/torch/vcodec.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/context.h"
#include "vcodec_encoder.cu.hxx"
#include "vcodec_util.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef __aarch64__

namespace abcdk
{
    namespace cuda
    {
        namespace vcodec
        {
            class encoder_aarch64 : public encoder
            {
            public:
                static encoder *create(CUcontext cuda_ctx)
                {
                    encoder *ctx = new encoder_aarch64(cuda_ctx);
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

                    delete (encoder_aarch64 *)ctx_p;
                }

            private:
                CUcontext m_gpu_ctx;

                nvmpictx *m_encoder;

                std::vector<uint8_t> m_ext_data;

            public:
                encoder_aarch64(CUcontext cuda_ctx)
                {
                    m_gpu_ctx = cuda_ctx;

                    m_encoder = NULL;
                }

                virtual ~encoder_aarch64()
                {
                    close();
                }

            protected:
                void GetSequenceParams(nvEncParam *param, std::vector<uint8_t> &seqParams)
                {
                    abcdk_torch_image_t *tmp;
                    std::vector<uint8_t> out;

                    /*创建一个图像，用于编码。*/
                    tmp = abcdk_torch_avframe_alloc(param->width, param->height, AV_PIX_FMT_YUV420P, 1);
                    if (!tmp)
                        return;

                    /*在100帧内编码器总能输出扩展数据了。*/
                    for (int i = 0; i < 100; i++)
                    {
                        if (encode(tmp, out) < 0)
                            break;

                        /*取第一个数据帧就可以了。*/
                        if (out.size() > 0)
                            break;
                    }

                    av_frame_free(&tmp);

                    if (out.size() <= 0)
                        return;

                    // H264 find IDR index 0x00 0x00 0x00 0x01 0x65
                    if (param->profile == FF_PROFILE_H264_MAIN)
                    {
                        int i = 0;
                        while (1)
                        {

                            if (out.size() < i + 5)
                                break;

                            if ((out[i] != 0 || out[i + 1] != 0 || out[i + 2] != 0 || out[i + 3] != 0x01 || out[i + 4] != 0x65)) // 找到关键帧开始的地方，在这之前的数据就是编码器扩展数据。
                                i++;
                            else
                                break;
                        }

                        if (i + 5 <= out.size())
                        {
                            seqParams.resize(i);
                            memcpy(seqParams.data(), out.data(), i);
                        }
                    }
                    // H265 find IDR index 0x00 0x00 0x00 0x01 0x26 0x01
                    else if (param->profile == FF_PROFILE_HEVC_MAIN)
                    {
                        int i = 0;
                        while (1)
                        {

                            if (out.size() < i + 6)
                                break;

                            if ((out[i] != 0 || out[i + 1] != 0 || out[i + 2] != 0 || out[i + 3] != 0x01 || out[i + 4] != 0x26 || out[i + 5] != 0x01)) // 找到关键帧开始的地方，在这之前的数据就是编码器扩展数据。
                                i++;
                            else
                                break;
                        }

                        if (i + 6 <= out.size())
                        {
                            seqParams.resize(i);
                            memcpy(seqParams.data(), out.data(), i);
                        }
                    }
                }

                int encode(const abcdk_torch_image_t *img, std::vector<uint8_t> &out)
                {
                    int frame_height[4] = {0};
                    nvFrame frame = {0};
                    nvPacket packet = {0};
                    int chk;

                    if (!m_encoder)
                        return -1;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    if (img)
                    {
                        frame.payload[0] = img->data[0];
                        frame.payload[1] = img->data[1];
                        frame.payload[2] = img->data[2];

                        abcdk_meida_image_fill_height(frame_height, img->height, img->pixfmt);
                        frame.payload_size[0] = img->stride[0] * frame_height[0];
                        frame.payload_size[1] = img->stride[1] * frame_height[1];
                        frame.payload_size[2] = img->stride[2] * frame_height[2];

                        frame.timestamp = img->pts;

                        chk = nvmpi_encoder_put_frame(m_encoder, &frame);
                        if (chk != 0)
                            return -1;
                    }

                    chk = nvmpi_encoder_get_packet(m_encoder, &packet);
                    if (chk != 0)
                        return -1;

                    if (packet.payload_size <= 0)
                        return 0;

                    out.clear();
                    out.insert(out.end(), &packet.payload[0], &packet.payload[packet.payload_size]);

                    return 1;
                }

            public:
                virtual void close()
                {
                    abcdk_torch_ctx_push(m_gpu_ctx);

                    if (m_encoder)
                        nvmpi_encoder_close(m_encoder);
                    m_encoder = NULL;

                    abcdk_torch_ctx_pop();

                }

                virtual int open(abcdk_torch_vcodec_param_t *param)
                {
                    int fps, width, height, flags;
                    nvCodingType nv_codecid;
                    nvEncParam nv_param = {0};

                    assert(param != NULL);

                    fps = param->fps_n/param->fps_d;
                    width = param->width;
                    height = param->height;
                    nv_codecid = (cudaVideoCodec)vcodec_to_nvcodec(param->format);

                    if (fps > 1000 || fps <= 0)
                        return -1;

                    if (width <= 0 || height <= 0)
                        return -1;

                    if (NV_VIDEO_CodingH264 != nv_codecid && NV_VIDEO_CodingHEVC != nv_codecid)
                        return -1;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    nv_param.width = width;
                    nv_param.height = height;
                    nv_param.bitrate = 4000000;
                    nv_param.mode_vbr = 0;
                    nv_param.idr_interval = fps;
                    nv_param.iframe_interval = fps;
                    nv_param.max_b_frames = 0;
                    nv_param.peak_bitrate = 0;
                    nv_param.fps_n = 1;
                    nv_param.fps_d = fps;
                    if (nv_codecid == NV_VIDEO_CodingH264)
                        nv_param.profile = FF_PROFILE_H264_MAIN;
                    if (nv_codecid == NV_VIDEO_CodingHEVC)
                        nv_param.profile = FF_PROFILE_HEVC_MAIN;
                    nv_param.level = -99;
                    nv_param.capture_num = 10;
                    nv_param.hw_preset_type = 3;
                    nv_param.insert_spspps_idr = param->insert_spspps_idr;

                    nv_param.qmax = -1;
                    nv_param.qmin = -1;

                    if (!param->insert_spspps_idr)
                    {
                        /*创建临时对象用于生成编码器扩展数据帧。*/
                        m_encoder = nvmpi_create_encoder(nv_codecid, &nv_param);
                        if (m_encoder)
                        {
                            GetSequenceParams(&param, m_ext_data);
                            nvmpi_encoder_close(m_encoder);
                            m_encoder = NULL;
                        }
                    }

                    m_encoder = nvmpi_create_encoder(nv_codecid, &nv_param);
                    if (!m_encoder)
                        return -1;

                    /*输出扩展数据帧。*/
                    if (ext_data.size() > 0)
                    {
                        param->ext_data = m_ext_data.data();
                        param->ext_size = m_ext_data.size();
                    }

                    return 0;
                }

                virtual int update(abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src)
                {
                    abcdk_torch_frame_t *tmp_src = NULL;
                    std::vector<uint8_t> out;
                    int chk;

                    assert(dst != NULL);

                    if (!m_encoder)
                        return -1;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    if (src)
                    {
                        if (src->format != (int)AV_PIX_FMT_YUV420P)
                        {
                            tmp_src = abcdk_torch_frame_create_cuda(src->width, src->height, AV_PIX_FMT_YUV420P, 1);
                            if (!tmp_src)
                                return -1;

                            chk = abcdk_torch_frame_convert_cuda(tmp_src, src); // 转换格式。

                            if (chk == 0)
                                chk = update(dst, tmp_src);

                            abcdk_torch_frame_free(&tmp_src);

                            return chk;
                        }

                        chk = encode(src, out);
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

                    chk = abcdk_torch_packet_reset(dst,out.size());
                    if(chk != 0)
                        return -1;

                    memcpy((*dst)->data, out.data(), out.size());

                    return 1;
                }
            };
        } // namespace vcodec
    } // namespace cuda
} // namespace abcdk

#endif // __aarch64__
#endif // __cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_VCODEC_ENCODER_AARCH64_HXX