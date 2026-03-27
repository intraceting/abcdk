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

#ifdef __XPU_NVIDIA__MMAPI__

#include "NvVideoEncoder.h"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace venc
        {
            typedef struct _metadata
            {
                context::metadata_t *rt_ctx;

                abcdk_xpu_vcodec_params_t params;

                NvVideoEncoder *cu_ctx;
                int output_plane_buf_ts;
            } metadata_t;

            static bool _encoder_capture_plane_cb(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer, NvBuffer *shared_buffer, void *arg)
            {
                metadata_t *ctx = (metadata_t *)arg;
                int chk;

                // write_encoder_output_frame(ctx->out_file, buffer);

                chk = ctx->cu_ctx->capture_plane.qBuffer(*v4l2_buf, NULL);
                if (chk != 0)
                    return false;

                /*到末尾了. */
                if (buffer->planes[0].bytesused == 0)
                    return false;

                return true;
            }

            static int _recv_packet(metadata_t *ctx, abcdk_object_t **dst, int64_t *ts)
            {
                return -1;
            }

            static int _send_frame(metadata_t *ctx, const image::metadata_t *src, int64_t ts)
            {
                struct v4l2_buffer v4l2_buf;
                struct v4l2_plane planes[MAX_PLANES];
                NvBuffer *buffer = NULL;
                int buf_idx;
                int chk;

                memset(&v4l2_buf, 0, sizeof(v4l2_buf));
                memset(planes, 0, sizeof(planes));

                if (ctx->cu_ctx->isInError())
                    return -1;

                if (ctx->output_plane_buf_ts < ctx->cu_ctx->output_plane.getNumBuffers())
                {
                    buf_idx = ctx->output_plane_buf_ts++;
                    buffer = ctx->cu_ctx->output_plane.getNthBuffer(buf_idx);

                    v4l2_buf.index = buf_idx;   // bind
                    v4l2_buf.m.planes = planes; // bind
                }
                else
                {
                    v4l2_buf.m.planes = planes; // bind

                    chk = ctx->cu_ctx->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
                    if (chk < 0)
                        return -2;
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

                ctx_p->cu_ctx->abort();

                common::util::delete_object(&ctx_p->cu_ctx);

                context::unref(&ctx_p->rt_ctx);

                common::util::delete_object(&ctx_p);
            }

            metadata_t *alloc()
            {
                metadata_t *ctx;

                ctx = new metadata_t;
                if (!ctx)
                    return NULL;

                ctx->cu_ctx = NULL;
                ctx->output_plane_buf_ts = 0;

                return ctx;
            }

            int setup(metadata_t *ctx, const abcdk_xpu_vcodec_params_t *params, context::metadata_t *rt_ctx)
            {
                int chk;

                ctx->rt_ctx = context::refer(rt_ctx);
                ctx->params = *params;

                if (!ctx->cu_ctx)
                {
                    ctx->cu_ctx = NvVideoEncoder::createVideoEncoder("nvenc");
                    if (!ctx->cu_ctx)
                        return -ENOMEM;
                }

                chk = ctx->cu_ctx->setCapturePlaneFormat(vcodec::local_to_nvcodec(ctx->params.format),
                                                         ctx->params.width, ctx->params.height, 8 * 1024 * 1024);
                if (chk != 0)
                    return -EINVAL;

                chk = ctx->cu_ctx->setOutputPlaneFormat(V4L2_PIX_FMT_RGB24, ctx->params.width, ctx->params.height);
                if (chk != 0)
                    return -EINVAL;

                chk = ctx->cu_ctx->setBitrate(ctx->params.bitrate);
                if (chk != 0)
                    return -EINVAL;

                if (ctx->params.format == ABCDK_XPU_VCODEC_ID_H264)
                    chk = ctx->cu_ctx->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_MAIN);
                else if (ctx->params.format == ABCDK_XPU_VCODEC_ID_H265)
                    chk = ctx->cu_ctx->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN);
                else 
                    chk = -1;
                if (chk != 0)
                    return -EINVAL;

                if (ctx->params.format == ABCDK_XPU_VCODEC_ID_H264)
                {
                    chk = ctx->cu_ctx->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_5_0);
                    if (chk != 0)
                        return -EINVAL;
                }

                chk = ctx->cu_ctx->setFrameRate(ctx->params.fps_n, ctx->params.fps_d);
                 if (chk != 0)
                    return -EINVAL;

                 chk = ctx->cu_ctx->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
                 if (chk != 0)
                     return -EINVAL;

                 chk = ctx->cu_ctx->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false);
                 if (chk != 0)
                     return -EINVAL;

                 chk = ctx->cu_ctx->output_plane.setStreamStatus(true);
                 if (chk != 0)
                     return -EINVAL;

                 chk = ctx->cu_ctx->capture_plane.setStreamStatus(true);
                 if (chk != 0)
                     return -EINVAL;

                 ctx->cu_ctx->capture_plane.setDQThreadCallback(_encoder_capture_plane_cb);
                 ctx->cu_ctx->capture_plane.startDQThread(ctx);

                 for (uint32_t i = 0; i < ctx->cu_ctx->capture_plane.getNumBuffers(); i++)
                 {
                     struct v4l2_buffer v4l2_buf;
                     struct v4l2_plane planes[MAX_PLANES];

                     memset(&v4l2_buf, 0, sizeof(v4l2_buf));
                     memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

                     v4l2_buf.index = i;
                     v4l2_buf.m.planes = planes;

                     chk = ctx->cu_ctx->capture_plane.qBuffer(v4l2_buf, NULL);
                     if (chk !=0 )
                        return -1;
                 }

                 return 0;
            }

            int get_params(metadata_t *ctx, abcdk_xpu_vcodec_params_t *params)
            {
                                
                return 0;
            }

            int recv_packet(metadata_t *ctx, abcdk_object_t **dst, int64_t *ts)
            {
                return -1;
            }

            int send_frame(metadata_t *ctx, const image::metadata_t *src, int64_t ts)
            {
                return -1;
            }
        } // namespace venc
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __XPU_NVIDIA__MMAPI__