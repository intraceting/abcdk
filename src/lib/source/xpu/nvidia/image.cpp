/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "image.hxx"

namespace abcdk_xpu
{

    namespace nvidia
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

                    memory::memset(ctx->data[i], 0, ctx->linesize[i] * real_height[i]);
                }
            }

            static void _free_buffer(void *opaque, uint8_t *data)
            {
                int in_host = (size_t)opaque;

                memory::free(data, in_host);
            }

            static int _get_buffer(AVFrame *ctx, int width, int height, AVPixelFormat pixfmt, int align, int in_host)
            {
                AVBufferRef *ff_buffer = NULL;
                int ff_stride[4] = {0};
                uint8_t *buf_ptr = NULL;
                int buf_size;
                int chk_size;

                if (abcdk_ffmpeg_image_fill_stride(ff_stride, width, pixfmt, align) <= 0)
                    return -EPERM;

                buf_size = abcdk_ffmpeg_image_get_size(ff_stride, height, pixfmt);
                if (buf_size <= 0)
                    return -EPERM;

                buf_ptr = memory::alloc_z<uint8_t>(buf_size ,in_host);
                if (buf_ptr == NULL)
                    return -ENOMEM;

                ff_buffer = av_buffer_create((uint8_t *)buf_ptr, buf_size, _free_buffer, (void*)(size_t)in_host, 0);
                if (!ff_buffer)
                {
                    memory::free(buf_ptr, in_host);
                    return -ENOMEM;
                }

                chk_size = abcdk_ffmpeg_image_fill_pointer(ctx->data, ff_stride, height, pixfmt, ff_buffer->data);
                assert(buf_size == chk_size);

                for (int i = 0; i < 4; i++)
                    ctx->linesize[i] = ff_stride[i];

                ctx->width = width;
                ctx->height = height;
                ctx->format = (int)pixfmt;
                ctx->buf[0] = ff_buffer; // bind.

                return 0;
            }

            int get_buffer(metadata_t *ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align, int in_host)
            {
                AVPixelFormat ff_pixfmt = pixfmt::local_to_ffmpeg(pixfmt);
                if (ff_pixfmt <= AV_PIX_FMT_NONE)
                    return -1;

                clear(ctx);//

                return _get_buffer(ctx, width, height, ff_pixfmt, align, in_host);
            }

            metadata_t *create(int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align, int in_host)
            {
                metadata_t *ctx = alloc();

                if (!ctx)
                    return NULL;

                int chk = get_buffer(ctx, width, height, pixfmt, align, in_host);
                if (chk != 0)
                {
                    free(&ctx);
                    return NULL;
                }

                return ctx;
            }

            int reset(metadata_t **ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align, int in_host)
            {
                int new_stride[4] = {0};
                int new_height[4] = {0};
                int old_height[4] = {0};
                metadata_t *ctx_p;
                int chk;

                if (!*ctx)
                {
                    *ctx = create(width, height, pixfmt, align, in_host);
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

                return get_buffer(ctx_p, width, height, pixfmt, align, in_host);
            }

            static int _copy(const AVFrame *src, int src_in_host, AVFrame *dst, int dst_in_host)
            {
                int real_stride[4] = {0};
                int real_height[4] = {0};
                int chk, chk_plane, chk_stride, chk_height;

                chk_stride = abcdk_ffmpeg_image_fill_stride(real_stride, src->width, (AVPixelFormat)src->format, 1);
                chk_height = abcdk_ffmpeg_image_fill_height(real_height, src->height, (AVPixelFormat)src->format);
                chk_plane = ABCDK_MIN(chk_stride, chk_height);

                for (int i = 0; i < chk_plane; i++)
                {
                    chk = memory::copy_2d(dst->data[i], dst->linesize[i], 0, 0, dst_in_host,
                                          src->data[i], src->linesize[i], 0, 0, src_in_host,
                                          real_stride[i], real_height[i]);
                    if (chk != 0)
                        return -1;
                }

                return 0;
            }
            
            int copy(const metadata_t *src, int src_in_host, metadata_t *dst, int dst_in_host)
            {
                return _copy(src, src_in_host, dst, dst_in_host);
            }

            int copy(const void *src_data, int src_linesize, int src_in_host, metadata_t *dst, int dst_plane, int dst_in_host)
            {
                int real_stride[4] = {0};
                int real_height[4] = {0};
                int chk, chk_stride, chk_height;

                chk_stride = abcdk_ffmpeg_image_fill_stride(real_stride, dst->width, (AVPixelFormat)dst->format, 1);
                chk_height = abcdk_ffmpeg_image_fill_height(real_height, dst->height, (AVPixelFormat)dst->format);
                chk = ABCDK_MIN(chk_stride, chk_height);

                assert(dst_plane < chk);

                memory::copy_2d(dst->data[dst_plane], dst->linesize[dst_plane], 0, 0, dst_in_host,
                                src_data, src_linesize, 0, 0, src_in_host,
                                real_stride[dst_plane], real_height[dst_plane]);

                return 0;
            }

            int copy(const metadata_t *src, int src_plane, int src_in_host, void *dst_data, int dst_linesize, int dst_in_host)
            {
                int real_stride[4] = {0};
                int real_height[4] = {0};
                int chk, chk_stride, chk_height;

                chk_stride = abcdk_ffmpeg_image_fill_stride(real_stride, src->width, (AVPixelFormat)src->format, 1);
                chk_height = abcdk_ffmpeg_image_fill_height(real_height, src->height, (AVPixelFormat)src->format);
                chk = ABCDK_MIN(chk_stride, chk_height);

                assert(src_plane < chk);

                memory::copy_2d(dst_data, dst_linesize, 0, 0, dst_in_host,
                                src->data[src_plane], src->linesize[src_plane], 0, 0, src_in_host,
                                real_stride[src_plane], real_height[src_plane]);

                return 0;
            }

            int copy(const cv::Mat &src, int src_in_host, metadata_t *dst, int dst_in_host)
            {
                AVFrame *ff_src;
                int chk;

                ff_src = av_frame_alloc();
                if (!ff_src)
                    return -ENOMEM;

                ff_src->width = src.cols;
                ff_src->height = src.rows;
                ff_src->format = dst->format;

                abcdk_ffmpeg_image_fill_stride(ff_src->linesize, ff_src->width, (AVPixelFormat)ff_src->format,1);//only step(1).
                abcdk_ffmpeg_image_fill_pointer(ff_src->data, ff_src->linesize, ff_src->height, (AVPixelFormat)ff_src->format, src.data);

                chk = _copy(ff_src, src_in_host, dst, dst_in_host);
                av_frame_free(&ff_src);

                return chk;
            }

            metadata_t *clone(const metadata_t *src, int src_in_host, int dst_align, int dst_in_host)
            {
                metadata_t *dst;
                int chk;

                dst = create(src->width,src->height, pixfmt::ffmpeg_to_local(src->format),dst_align,dst_in_host);
                if(!dst)
                    return NULL;

                chk = copy(src,src_in_host,dst,dst_in_host);
                if(chk != 0)
                {
                    free(&dst);
                    return NULL;
                }

                return dst;
            }

            metadata_t *clone(abcdk_xpu_pixfmt_t pixfmt,const cv::Mat &src, int src_in_host, int dst_align, int dst_in_host)
            {
                metadata_t *dst;
                int chk;

                dst = create(src.cols,src.rows, pixfmt,dst_align,dst_in_host);
                if(!dst)
                    return NULL;

                chk = copy(src,src_in_host,dst,dst_in_host);
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

                    copy(src_data[i], src_linesize[i], 1, dst, i, 0);
                }

                return 0;
            }

            int download(const metadata_t *src, uint8_t *dst_data[4], int dst_linesize[4])
            {
                for (int i = 0; i < 4; i++)
                {
                    if (dst_data[i] == NULL || dst_linesize[i] <= 0)
                        return 0;

                    copy(src, i, 0, dst_data[i], dst_linesize[i], 1);
                }

                return 0;
            }

        } // namespace image
    } // namespace nvidia

} // namespace abcdk_xpu