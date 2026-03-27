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
#include "abcdk/util/queue.h"
#include "abcdk/util/h2645.h"
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

                std::vector<uint8_t> ext_data;

                NvVideoEncoder *cu_ctx;
                int output_buf_ts;

                abcdk_queue_t *capture_queue_ctx;
                int capture_eof;
            } metadata_t;

            static void _capture_queue_destroy_cb(void *msg)
            {
                abcdk_object_unref((abcdk_object_t**)&msg);
            }

            static bool _capture_write_cb(struct v4l2_buffer *v4l2_buf, NvBuffer *buffer, NvBuffer *shared_buffer, void *arg)
            {
                metadata_t *ctx = (metadata_t *)arg;
                abcdk_object_t* pkt = NULL;
                int chk;

                /*到末尾了. */
                if (buffer->planes[0].bytesused == 0)
                {
                    chk = ctx->cu_ctx->capture_plane.qBuffer(*v4l2_buf, NULL);
                    assert(chk == 0);

                    ctx->capture_eof = 1;
                    return false;
                }

                if (ctx->ext_data.size() <= 0)
                {
                    chk = abcdk_h2645_extract_sei(buffer->planes[0].data, buffer->planes[0].bytesused, ctx->params.format == ABCDK_XPU_VCODEC_ID_H264);
                    if (chk > 0)
                    {
                        ctx->ext_data.insert(ctx->ext_data.end(), buffer->planes[0].data, buffer->planes[0].data + chk);
                    }
                }

                size_t sizes[2] = {buffer->planes[0].bytesused,sizeof(int64_t)};
                pkt = abcdk_object_alloc(sizes,2,0);
                if(!pkt)
                {
                    chk = ctx->cu_ctx->capture_plane.qBuffer(*v4l2_buf, NULL);
                    assert(chk == 0);

                    ctx->capture_eof = 1;
                    return false;
                }

                memcpy(pkt->pptrs[0], buffer->planes[0].data, buffer->planes[0].bytesused);
                *((int64_t *)pkt->pptrs[1]) = v4l2_buf->timestamp.tv_sec * 1000000 + v4l2_buf->timestamp.tv_usec;

                abcdk_queue_lock(ctx->capture_queue_ctx);
                abcdk_queue_push(ctx->capture_queue_ctx, pkt);
                abcdk_queue_unlock(ctx->capture_queue_ctx);

                chk = ctx->cu_ctx->capture_plane.qBuffer(*v4l2_buf, NULL);
                if (chk != 0)
                {
                    ctx->capture_eof = 1;
                    return false;
                }
                
                return true;
            }

            static int _recv_packet(metadata_t *ctx, abcdk_object_t **dst, int64_t *ts)
            {
                abcdk_object_t *pkt;
                abcdk_object_unref(dst);

                abcdk_queue_lock(ctx->capture_queue_ctx);
                pkt = (abcdk_object_t *)abcdk_queue_pop(ctx->capture_queue_ctx);
                abcdk_queue_unlock(ctx->capture_queue_ctx);

                if (pkt)
                {
                    *dst = abcdk_object_copyfrom(pkt->pptrs[0], pkt->sizes[0]);
                    *ts = *((int64_t *)pkt->pptrs[1]);

                    abcdk_object_unref(&pkt);
                    return 1;
                }

                if (ctx->capture_eof)
                    return -1;

                return 0;
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

                if (ctx->output_buf_ts < ctx->cu_ctx->output_plane.getNumBuffers())
                {
                    buf_idx = ctx->output_buf_ts++;
                    buffer = ctx->cu_ctx->output_plane.getNthBuffer(buf_idx);

                    v4l2_buf.index = buf_idx;   // bind
                    v4l2_buf.m.planes = planes; // bind
                }
                else
                {
                    v4l2_buf.m.planes = planes; // bind

                    chk = ctx->cu_ctx->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
                    if (chk != 0)
                        return -2;
                }

                if(!src)
                {
                    v4l2_buf.m.planes[0].bytesused = 0; // 空包
                    v4l2_buf.flags |= V4L2_BUF_FLAG_LAST; // 最后一包.
                }
                else
                {
                    chk = image::copy(src,0,buffer,1);
                    if (chk != 0)
                        return -4;

                    v4l2_buf.timestamp.tv_sec = ts / 1000000;
                    v4l2_buf.timestamp.tv_usec = ts % 1000000;
                    v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
                }

                chk = ctx->cu_ctx->output_plane.qBuffer(v4l2_buf, NULL);
                if (chk < 0)
                    return -3;

                return -1;
            }

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                ctx_p->cu_ctx->capture_plane.stopDQThread();
                ctx_p->cu_ctx->capture_plane.waitForDQThread(2000);
                ctx_p->cu_ctx->abort();

                abcdk_queue_free(&ctx_p->capture_queue_ctx);

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
                ctx->output_buf_ts = 0;

                ctx->capture_queue_ctx = abcdk_queue_alloc(_capture_queue_destroy_cb);
                ctx->capture_eof = 0;

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

                if (ctx->params.fps_n <= 0 || ctx->params.fps_d <= 0)
                    return -EINVAL;

                if (ctx->params.width <= 0 || ctx->params.height <= 0)
                    return -EINVAL;

                if (ctx->params.format == ABCDK_XPU_VCODEC_ID_H264 && ctx->params.format == ABCDK_XPU_VCODEC_ID_H265)
                    return -EINVAL;

                chk = ctx->cu_ctx->setCapturePlaneFormat(vcodec::local_to_nvcodec(ctx->params.format),
                                                         ctx->params.width, ctx->params.height, 8 * 1024 * 1024);
                if (chk != 0)
                    return -EINVAL;

                chk = ctx->cu_ctx->setOutputPlaneFormat(V4L2_PIX_FMT_RGB32, ctx->params.width, ctx->params.height);
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

                 ctx->cu_ctx->capture_plane.setDQThreadCallback(_capture_write_cb);
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

                 for (int i = 0; i < 10; i++)
                 {
                     image::metadata_t *img = image::create(ctx->params.width, ctx->params.height, ABCDK_XPU_PIXFMT_RGB32, 16, 0);
                     if (!img)
                         return -2;

                     send_frame(ctx, img, 0);
                     abcdk_nanosleep(200 * 1000000ULL);//sleep 200mills.

                     if (ctx->ext_data.size() > 0)
                         break;
                 }

                 if (ctx->ext_data.size() > 0)
                 {
                     ctx->params.ext_data = ctx->ext_data.data();
                     ctx->params.ext_size = ctx->ext_data.size();
                 }
                 else
                 {
                     abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("未能获取SEI信息."));
                 }

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

#endif // #ifdef __XPU_NVIDIA__MMAPI__