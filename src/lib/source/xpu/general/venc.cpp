/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/atomic.h"
#include "abcdk/util/object.h"
#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/ffmpeg/ffmpeg.h"
#include "abcdk/ffmpeg/encoder.h"
#include "abcdk/xpu/image.h"
#include "venc.hxx"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace venc
        {
            typedef struct _metadata
            {
                context::metadata_t *rt_ctx;

                abcdk_xpu_vcodec_params_t params;

                std::vector<uint8_t> ext_data;

                abcdk_ffmpeg_encoder_t *ff_ctx;
                AVCodecParameters *ff_par;
                AVRational time_base;
                AVRational frame_rate;

            } metadata_t;

            static int _init(metadata_t *ctx)
            {
                int chk;

                ctx->ff_par = avcodec_parameters_alloc();
                if (!ctx->ff_par)
                    return -1;

                ctx->ff_par->codec_type = AVMEDIA_TYPE_VIDEO;
                ctx->ff_par->codec_id = util::local_to_ffmpeg(ctx->params.format);
                ctx->ff_par->width = ctx->params.width;
                ctx->ff_par->height = ctx->params.height;
                ctx->ff_par->bit_rate = ctx->params.bitrate;
                ctx->ff_par->profile = ctx->params.profile;
                ctx->ff_par->level = ctx->params.level;
                ctx->ff_par->video_delay = ctx->params.max_b_frames;
                ctx->ff_par->extradata = (uint8_t *)ctx->params.ext_data;
                ctx->ff_par->extradata_size = ctx->params.ext_size;
                ctx->ff_par->format = AV_PIX_FMT_YUV420P;

                ctx->ff_ctx = abcdk_ffmpeg_encoder_alloc3(ctx->ff_par->codec_id);
                if (!ctx->ff_ctx)
                    return -1;

                ctx->time_base.num = ctx->params.fps_d;
                ctx->time_base.den = ctx->params.fps_n;

                ctx->frame_rate.num = ctx->params.fps_n;
                ctx->frame_rate.den = ctx->params.fps_d;

                chk = abcdk_ffmpeg_encoder_init(ctx->ff_ctx, ctx->ff_par, &ctx->time_base, &ctx->frame_rate);
                if (chk != 0)
                    return chk;

                chk = abcdk_ffmpeg_encoder_open(ctx->ff_ctx, NULL);
                if (chk != 0)
                    return chk;

                abcdk_ffmpeg_encoder_get_param(ctx->ff_ctx, ctx->ff_par);

                ctx->ext_data.clear();
                ctx->ext_data.insert(ctx->ext_data.end(), &ctx->ff_par->extradata[0], &ctx->ff_par->extradata[ctx->ff_par->extradata_size]);

                // 复制指针和长度.
                ctx->params.ext_data = ctx->ext_data.data();
                ctx->params.ext_size = ctx->ext_data.size();

                return 0;
            }

            static int _recv_packet(metadata_t *ctx, abcdk_object_t **dst, int64_t *ts)
            {
                AVPacket tmp_dst;
                int chk;

                av_init_packet(&tmp_dst);

                chk = abcdk_ffmpeg_encoder_recv(ctx->ff_ctx, &tmp_dst);
                if (chk < 0)
                    return -1;
                else if (chk == 0)
                    return 0;

                abcdk_object_unref(dst);
                *dst = abcdk_object_copyfrom(tmp_dst.data, tmp_dst.size);
                *ts = tmp_dst.pts;

                av_packet_unref(&tmp_dst); // Don't forget.

                return 1;
            }

            static int _send_frame(metadata_t *ctx, const image::metadata_t *src, int64_t ts)
            {
                AVFrame *tmp_src;
                int chk;

                if (src)
                {
                    tmp_src = av_frame_alloc();
                    if (!tmp_src)
                        return -1;

                    tmp_src->width = src->width;
                    tmp_src->height = src->height;
                    tmp_src->format = src->format;
                    tmp_src->pts = ts;

                    for (int i = 0; i < 4; i++)
                    {
                        tmp_src->data[i] = src->data[i];
                        tmp_src->linesize[i] = src->linesize[i];
                    }

                    chk = abcdk_ffmpeg_encoder_send(ctx->ff_ctx, tmp_src);
                    av_frame_free(&tmp_src); // Don't forget.

                    if (chk < 0)
                        return -1;
                }
                else
                {
                    chk = abcdk_ffmpeg_encoder_send(ctx->ff_ctx, NULL);
                    if (chk < 0)
                        return -1;
                }

                return 1;
            }

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                abcdk_ffmpeg_encoder_free(&ctx_p->ff_ctx);

                ctx_p->ff_par->extradata = NULL; // must be clear.
                ctx_p->ff_par->extradata_size = 0;
                avcodec_parameters_free(&ctx_p->ff_par);

                context::unref(&ctx_p->rt_ctx);

                delete ctx_p;
            }

            metadata_t *alloc()
            {
                metadata_t *ctx;

                ctx = new metadata_t;
                if (!ctx)
                    return NULL;

                ctx->rt_ctx = NULL;
                ctx->ff_ctx = NULL;
                ctx->ff_par = NULL;

                return ctx;
            }

            int setup(metadata_t *ctx, const abcdk_xpu_vcodec_params_t *params, context::metadata_t *rt_ctx)
            {
                int chk;

                ctx->rt_ctx = context::refer(rt_ctx);
                ctx->params = *params;

                ctx->params.ext_data = NULL;
                ctx->params.ext_size = 0;

                chk = _init(ctx);
                if (chk != 0)
                    return chk;

                return 0;
            }

            int get_params(metadata_t *ctx, abcdk_xpu_vcodec_params_t *params)
            {
                *params = ctx->params;

                params->ext_data = ctx->ext_data.data();
                params->ext_size = ctx->ext_data.size();

                return 0;
            }

            int recv_packet(metadata_t *ctx, abcdk_object_t **dst, int64_t *ts)
            {
                return _recv_packet(ctx,dst,ts);
            }

            int send_frame(metadata_t *ctx, const image::metadata_t *src, int64_t ts)
            {
                image::metadata_t *tmp_src;
                int chk;

                if (src != NULL && src->format != AV_PIX_FMT_YUV420P)
                {
                    tmp_src = image::create(src->width, src->height, ABCDK_XPU_PIXFMT_YUV420P, 16);
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
    } // namespace general
} // namespace abcdk_xpu
