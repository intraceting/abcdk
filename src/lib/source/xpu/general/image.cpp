/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/general.h"
#include "image.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace image
        {

            void free(metadata_t **ctx)
            {
                common::image::free(ctx);
            }

            metadata_t *alloc()
            {
                return common::image::alloc();
            }

            void clear(metadata_t *ctx)
            {
                common::image::clear(ctx);
            }

            void zero(metadata_t *ctx)
            {
                int real_height[4] = {0};
                int chk_height;

                chk_height = abcdk_ffmpeg_image_fill_height(real_height, ctx->height, (AVPixelFormat)ctx->format);

                for (int i = 0; i < 4; i++)
                {
                    if (ctx->data[i] == NULL || ctx->linesize[i] <= 0)
                        break;

                    memset(ctx->data[i], 0, ctx->linesize[i] * real_height[i]);
                }
            }

            int _get_buffer(AVFrame *ctx, int width, int height, AVPixelFormat pixfmt, int align)
            {
                ctx->width = width;
                ctx->height = height;
                ctx->format = (int)pixfmt;

                return common::image::get_buffer(ctx, align);
            }

            int get_buffer(metadata_t *ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align)
            {
                AVPixelFormat ff_pixfmt = pixfmt::local_to_ffmpeg(pixfmt);
                if (ff_pixfmt <= AV_PIX_FMT_NONE)
                    return -1;

                clear(ctx);//

                return _get_buffer(ctx, width, height, ff_pixfmt, align);
            }

            metadata_t *create(int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align)
            {
                metadata_t *ctx = alloc();

                if (!ctx)
                    return NULL;

                int chk = get_buffer(ctx, width, height, pixfmt, align);
                if (chk != 0)
                {
                    free(&ctx);
                    return NULL;
                }

                return ctx;
            }

            int reset(metadata_t **ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align)
            {
                int new_stride[4] = {0};
                int new_height[4] = {0};
                int old_height[4] = {0};
                metadata_t *ctx_p;
                int chk;

                if (!*ctx)
                {
                    *ctx = create(width, height, pixfmt, align);
                    if (!*ctx)
                        return -ENOMEM;

                    return 0;
                }

                ctx_p = *ctx;

                if (width != ctx_p->width || height != ctx_p->height || pixfmt::local_to_ffmpeg(pixfmt) != ctx_p->format)
                    goto NEW;

                abcdk_ffmpeg_image_fill_stride(new_stride, width, pixfmt::local_to_ffmpeg(pixfmt), align);
                abcdk_ffmpeg_image_fill_height(new_height, height, pixfmt::local_to_ffmpeg(pixfmt));

                abcdk_ffmpeg_image_fill_height(old_height, ctx_p->height, (AVPixelFormat)ctx_p->format);

                for (int i = 0; i < 4; i++)
                {
                    if (new_stride[i] <= 0 || ctx_p->linesize[i] <= 0)
                        break;

                    if (new_stride[i] != ctx_p->linesize[i])
                        goto NEW;
                }

                for (int i = 0; i < 4; i++)
                {
                    if (new_height[i] <= 0 || old_height[i] <= 0)
                        break;

                    if (new_height[i] != old_height[i])
                        goto NEW;
                }


                return 0;

            NEW:

                return get_buffer(ctx_p, width, height, pixfmt, align);
            }
            
            int copy(const metadata_t *src, metadata_t *dst)
            {
                return common::image::copy(src, dst);
            }

            int copy(const void *src_data, int src_linesize, metadata_t *dst, int dst_plane)
            {
                int real_stride[4] = {0};
                int real_height[4] = {0};
                int chk, chk_stride, chk_height;

                chk_stride = abcdk_ffmpeg_image_fill_stride(real_stride, dst->width, (AVPixelFormat)dst->format, 1);
                chk_height = abcdk_ffmpeg_image_fill_height(real_height, dst->height, (AVPixelFormat)dst->format);
                chk = ABCDK_MIN(chk_stride, chk_height);

                assert(dst_plane < chk);

                abcdk_memcpy_2d(dst->data[dst_plane], dst->linesize[dst_plane], 0, 0,
                                src_data, src_linesize, 0, 0,
                                real_stride[dst_plane], real_height[dst_plane]);

                return 0;
            }

            int copy(const metadata_t *src, int src_plane, void *dst_data, int dst_linesize)
            {
                int real_stride[4] = {0};
                int real_height[4] = {0};
                int chk, chk_stride, chk_height;

                chk_stride = abcdk_ffmpeg_image_fill_stride(real_stride, src->width, (AVPixelFormat)src->format, 1);
                chk_height = abcdk_ffmpeg_image_fill_height(real_height, src->height, (AVPixelFormat)src->format);
                chk = ABCDK_MIN(chk_stride, chk_height);

                assert(src_plane < chk);

                abcdk_memcpy_2d(dst_data, dst_linesize, 0, 0,
                                src->data[src_plane], src->linesize[src_plane], 0, 0,
                                real_stride[src_plane], real_height[src_plane]);

                return 0;
            }

            int copy(const cv::Mat &src, metadata_t *dst)
            {
                return common::image::copy(src,dst);
            }
           
            metadata_t *clone(const metadata_t *src, int dst_align)
            {
                metadata_t *dst;
                int chk;

                dst = create(src->width,src->height,pixfmt::ffmpeg_to_local(src->format),dst_align);
                if(!dst)
                    return NULL;

                chk = copy(src,dst);
                if(chk != 0)
                {
                    image::free(&dst);
                    return dst;
                }

                return dst;
            }

            metadata_t *clone(abcdk_xpu_pixfmt_t pixfmt, const cv::Mat &src, int dst_align)
            {
                metadata_t *dst;
                int chk;

                dst = create(src.cols,src.rows, pixfmt,dst_align);
                if(!dst)
                    return NULL;

                chk = copy(src,dst);
                if(chk != 0)
                {
                    free(&dst);
                    return NULL;
                }

                return dst;
            }

            int get_width(const metadata_t *src)
            {
                return src->width;
            }

            int get_height(const metadata_t *src)
            {
                return src->height;
            }

            abcdk_xpu_pixfmt_t get_pixfmt(const metadata_t *src)
            {
                return pixfmt::ffmpeg_to_local(src->format);
            }

            int upload(const uint8_t *src_data[4], const int src_linesize[4], metadata_t *dst)
            {
                for (int i = 0; i < 4; i++)
                {
                    if (src_data[i] == NULL || src_linesize[i] <= 0)
                        return 0;

                    copy(src_data[i], src_linesize[i], dst, i);
                }

                return 0;
            }

            int download(const metadata_t *src, uint8_t *dst_data[4], int dst_linesize[4])
            {
                for (int i = 0; i < 4; i++)
                {
                    if (dst_data[i] == NULL || dst_linesize[i] <= 0)
                        return 0;

                    copy(src, i, dst_data[i], dst_linesize[i]);
                }

                return 0;
            }

        } // namespace image
    } // namespace general
} // namespace abcdk_xpu