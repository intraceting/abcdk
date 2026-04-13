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
#include "abcdk/ffmpeg/decoder.h"
#include "abcdk/xpu/image.h"
#include "vdec.hxx"
#include "util.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace vdec
        {
            typedef struct _metadata
            {
                context::metadata_t *rt_ctx;

                abcdk_xpu_vcodec_params_t params;

                std::vector<uint8_t> ext_data;

                abcdk_ffmpeg_decoder_t *ff_ctx;
                AVCodecParameters *ff_par;

            } metadata_t;

            static int _init(metadata_t *ctx)
            {
                int chk;

                ctx->ff_par = avcodec_parameters_alloc();
                if (!ctx->ff_par)
                    return -1;

                ctx->ff_par->codec_type = AVMEDIA_TYPE_VIDEO;
                ctx->ff_par->codec_id = util::local_to_ffmpeg(ctx->params.format);
                // ctx->ff_par->width = ctx->params.width;
                // ctx->ff_par->height = ctx->params.height;
                // ctx->ff_par->bit_rate = ctx->params.bitrate;
                // ctx->ff_par->video_delay = ctx->params.max_b_frames;
                ctx->ff_par->extradata = (uint8_t *)ctx->params.ext_data;
                ctx->ff_par->extradata_size = ctx->params.ext_size;
                ctx->ff_par->format = AV_PIX_FMT_NV12;

                ctx->ff_ctx = abcdk_ffmpeg_decoder_alloc3(ctx->ff_par->codec_id);
                if (!ctx->ff_ctx)
                    return -1;

                chk = abcdk_ffmpeg_decoder_init(ctx->ff_ctx, ctx->ff_par);
                if (chk != 0)
                    return chk;

                chk = abcdk_ffmpeg_decoder_open(ctx->ff_ctx, NULL);
                if (chk != 0)
                    return chk;

                return 0;
            }

            static int _send_packet(metadata_t *ctx, const void *src_data, size_t src_size, int64_t ts)
            {
                AVPacket tmp_src;
                int chk;

                av_init_packet(&tmp_src);

                if (src_data != NULL && src_size > 0)
                {
                    tmp_src.data = (uint8_t *)src_data;
                    tmp_src.size = src_size;
                    tmp_src.pts = ts;

                    chk = abcdk_ffmpeg_decoder_send(ctx->ff_ctx, &tmp_src);
                    if (chk != 0)
                        return -1;
                }
                else
                {
                    chk = abcdk_ffmpeg_decoder_send(ctx->ff_ctx, NULL);
                    if (chk != 0)
                        return -1;
                }

                return 1;
            }


            static int _recv_frame(metadata_t *ctx, image::metadata_t **dst, int64_t *ts)
            {
                int chk;

                if (!*dst)
                    *dst = image::alloc();

                if (!*dst)
                    return -1;

                chk = abcdk_ffmpeg_decoder_recv(ctx->ff_ctx, *dst);
                if (chk < 0)
                    return -1;
                else if (chk == 0)
                    return 0;

                *ts = (*dst)->pts;
                return 1;
            }

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                abcdk_ffmpeg_decoder_free(&ctx_p->ff_ctx);

                ctx_p->ff_par->extradata = NULL;//must be clear.
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

                ctx->ext_data.resize(params->ext_size);
                memcpy(ctx->ext_data.data(), params->ext_data, params->ext_size);

                // 复制指针和长度.
                ctx->params.ext_data = ctx->ext_data.data();
                ctx->params.ext_size = ctx->ext_data.size();

                chk = _init(ctx);
                if (chk != 0)
                    return chk;

                return 0;
            }

            int send_packet(metadata_t *ctx, const void *src_data, size_t src_size, int64_t ts)
            {
                return _send_packet(ctx,src_data,src_size,ts);
            }

            int recv_frame(metadata_t *ctx, image::metadata_t **dst, int64_t *ts)
            {
                return _recv_frame(ctx,dst,ts);
            }
        } // namespace vdec
    } // namespace general
} // namespace abcdk_xpu
