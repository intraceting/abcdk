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

#ifdef __XPU_NVIDIA__MMAPI__

#include "NvVideoDecoder.h"
#include "NvBufSurface.h"

namespace abcdk_xpu
{
    namespace nvidia
    {
        namespace vdec
        {
            typedef struct _metadata
            {
                context::metadata_t *rt_ctx;

                abcdk_xpu_vcodec_params_t params;

                NvVideoDecoder *cu_ctx;
                int output_plane_buf_ts;

                int capture_info_ok;
                struct v4l2_format capture_format;
                struct v4l2_crop capture_crop;
                int capture_dma_fd;

            } metadata_t;

            static void _revoke_output_buffer(metadata_t *ctx)
            {
                struct v4l2_buffer v4l2_buf;
                struct v4l2_plane planes[MAX_PLANES];

                for (int i = 0; i < ctx->cu_ctx->output_plane.getNumBuffers(); i++)
                {
                    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
                    memset(planes, 0, sizeof(planes));

                    v4l2_buf.m.planes = planes;
                    int chk = ctx->cu_ctx->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
                    ABCDK_UNUSED(chk);
                }
            }

            static void _revoke_capture_buffer(metadata_t *ctx)
            {
                struct v4l2_buffer v4l2_buf;
                struct v4l2_plane planes[MAX_PLANES];

                for (int i = 0; i < ctx->cu_ctx->capture_plane.getNumBuffers(); i++)
                {
                    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
                    memset(planes, 0, sizeof(planes));

                    v4l2_buf.m.planes = planes;
                    int chk = ctx->cu_ctx->capture_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
                    ABCDK_UNUSED(chk);
                }
            }

            static int _send_packet(metadata_t *ctx, const void *src_data, size_t src_size, int64_t ts)
            {
                struct v4l2_buffer v4l2_buf;
                struct v4l2_plane planes[MAX_PLANES];
                NvBuffer *buffer = NULL;
                uint32_t buf_idx;
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

                if (!src_data || src_size <= 0)
                {
                    v4l2_buf.m.planes[0].bytesused = 0; // 空包
                    v4l2_buf.flags |= V4L2_BUF_FLAG_LAST; // 最后一包.

                    chk = ctx->cu_ctx->output_plane.qBuffer(v4l2_buf, NULL);
                    if (chk < 0)
                        return -3;

                    return 1;
                }
                else
                {
                    assert(buffer->planes[0].length >= src_size);

                    memcpy((char *)buffer->planes[0].data, src_data, src_size);
                    buffer->planes[0].bytesused = src_size;

                    v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

                    v4l2_buf.timestamp.tv_sec = ts / 1000000;
                    v4l2_buf.timestamp.tv_usec = ts % 1000000;
                    v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;

                    chk = ctx->cu_ctx->output_plane.qBuffer(v4l2_buf, NULL);
                    if (chk < 0)
                        return -3;

                    return 1;
                }
            }

            static int _recv_frame(metadata_t *ctx, image::metadata_t **dst, int64_t *ts)
            {
                NvBufSurf::NvCommonTransformParams transform_params = {0};
                NvBufSurf::NvCommonAllocateParams params = {0};
                int min_dec_capture_buffers;
                struct v4l2_event ev;
                struct v4l2_buffer v4l2_buf;
                struct v4l2_plane planes[MAX_PLANES];
                NvBuffer *buffer = NULL;
                NvBufSurface *dma_surf = NULL;
                int chk;

                if (ctx->cu_ctx->isInError())
                    return -1;

                if (!ctx->capture_info_ok)
                {
                    chk = ctx->cu_ctx->dqEvent(ev, 10);
                    if (chk != 0)
                        return (errno == EAGAIN ? 0 : -1);

                    chk = ctx->cu_ctx->capture_plane.getFormat(ctx->capture_format);
                    if (chk != 0)
                        return -1;

                    chk = ctx->cu_ctx->capture_plane.getCrop(ctx->capture_crop);
                    if (chk != 0)
                        return -2;

                    ctx->cu_ctx->capture_plane.deinitPlane();

                    chk = ctx->cu_ctx->setCapturePlaneFormat(ctx->capture_format.fmt.pix_mp.pixelformat,
                                                             ctx->capture_format.fmt.pix_mp.width, ctx->capture_format.fmt.pix_mp.height);
                    if (chk != 0)
                        return -4;

                    chk = ctx->cu_ctx->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
                    if (chk != 0)
                        return -5;

                    chk = ctx->cu_ctx->capture_plane.setupPlane(V4L2_MEMORY_MMAP, min_dec_capture_buffers + 5, false, false);
                    if (chk != 0)
                        return -6;

                    chk = ctx->cu_ctx->capture_plane.setStreamStatus(true);
                    if (chk != 0)
                        return -7;

                    /* 所有缓存入队列. */
                    for (uint32_t i = 0; i < ctx->cu_ctx->capture_plane.getNumBuffers(); i++)
                    {
                        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
                        memset(planes, 0, sizeof(planes));

                        v4l2_buf.index = i;
                        v4l2_buf.m.planes = planes;
                        chk = ctx->cu_ctx->capture_plane.qBuffer(v4l2_buf, NULL);
                        if (chk != 0)
                            return -8;
                    }

                    params.memType = NVBUF_MEM_SURFACE_ARRAY;
                    params.width = ctx->capture_crop.c.width;
                    params.height = ctx->capture_crop.c.height;
                    params.layout = NVBUF_LAYOUT_PITCH;
                    params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
                    params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

                    chk = NvBufSurf::NvAllocate(&params, 1, &ctx->capture_dma_fd);
                    if (chk != 0)
                        return -3;

                    ctx->capture_info_ok = 1;
                }

                memset(&v4l2_buf, 0, sizeof(v4l2_buf));
                memset(planes, 0, sizeof(planes));
                v4l2_buf.m.planes = planes;

                chk = ctx->cu_ctx->capture_plane.dqBuffer(v4l2_buf, &buffer, NULL, 0);
                if (chk != 0)
                    return (errno == EAGAIN ? 0 : -1);
                
                transform_params.src_top = 0;
                transform_params.src_left = 0;
                transform_params.src_width = ctx->capture_crop.c.width;
                transform_params.src_height = ctx->capture_crop.c.height;
                transform_params.dst_top = 0;
                transform_params.dst_left = 0;
                transform_params.dst_width = ctx->capture_crop.c.width;
                transform_params.dst_height = ctx->capture_crop.c.height;
                transform_params.flag = NVBUFSURF_TRANSFORM_FILTER;
                transform_params.flip = NvBufSurfTransform_None;
                transform_params.filter = NvBufSurfTransformInter_Nearest;

                chk = NvBufSurf::NvTransform(&transform_params, buffer->planes[0].fd, ctx->capture_dma_fd);
                if (chk != 0)
                    return -2;

                chk = NvBufSurfaceFromFd(ctx->capture_dma_fd, (void **)&dma_surf);
                if (chk != 0)
                    return -3;

                chk = image::reset(dst, ctx->capture_crop.c.width, ctx->capture_crop.c.height, ABCDK_XPU_PIXFMT_YUV420P, 16, 0);
                if (chk != 0)
                    return -4;

                chk = image::copy(dma_surf, 1, *dst, 0);
                if (chk != 0)
                    return -5;

                *ts = v4l2_buf.timestamp.tv_sec * 1000000 + v4l2_buf.timestamp.tv_usec;//PTS

                chk = ctx->cu_ctx->capture_plane.qBuffer(v4l2_buf, NULL);
                if (chk != 0)
                    return -6;

                return 1;
            }

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                if (ctx_p->capture_dma_fd > 0)
                    NvBufSurf::NvDestroy(ctx_p->capture_dma_fd);

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
                    ctx->cu_ctx = NvVideoDecoder::createVideoDecoder("nvdec");
                    if (!ctx->cu_ctx)
                        return -ENOMEM;
                }

                chk = ctx->cu_ctx->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
                if (chk != 0)
                    return -EINVAL;

                chk = ctx->cu_ctx->setOutputPlaneFormat(vcodec::local_to_nvcodec(ctx->params.format), 8 * 1024 * 1024);
                if (chk != 0)
                    return -EINVAL;

                chk = ctx->cu_ctx->setFrameInputMode(1);
                if (chk != 0)
                    return -EINVAL;

                chk = ctx->cu_ctx->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
                if (chk != 0)
                    return -EINVAL;

                chk = ctx->cu_ctx->output_plane.setStreamStatus(true);
                if (chk != 0)
                    return -EINVAL;

                return 0;
            }

            int send_packet(metadata_t *ctx, const void *src_data, size_t src_size, int64_t ts)
            {
                return _send_packet(ctx, src_data, src_size, ts);
            }

            int recv_frame(metadata_t *ctx, image::metadata_t **dst, int64_t *ts)
            {
                return _recv_frame(ctx, dst, ts);
            }

        } // namespace vdec
    } // namespace nvidia
} // namespace abcdk_xpu

#endif // #ifdef __XPU_NVIDIA__MMAPI__