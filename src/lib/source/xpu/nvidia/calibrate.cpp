/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "calibrate.hxx"

namespace abcdk_xpu
{
    namespace nvidia
    {
        namespace calibrate
        {
            typedef struct _metadata
            {
                std::shared_ptr<common::calibrate> co_ctx;
                image::metadata_t *warper_xmap;
                image::metadata_t *warper_ymap;
            } metadata_t;

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                image::free(&ctx_p->warper_xmap);
                image::free(&ctx_p->warper_ymap);

                delete ctx_p;
            }

            metadata_t *alloc()
            {
                metadata_t *ctx;

                ctx = new metadata_t;
                if (!ctx)
                    return NULL;

                ctx->co_ctx = common::calibrate::create();
                ctx->warper_xmap = NULL;
                ctx->warper_ymap = NULL;

                return ctx;
            }

            void setup(metadata_t *ctx, int board_cols, int board_rows, float grid_width, float grid_height)
            {
                ctx->co_ctx->setup(cv::Size(board_cols, board_rows), cv::Size2f(grid_width, grid_height));
            }

            static int _detect_corners(metadata_t *ctx, const image::metadata_t *src, int win_width, int win_height)
            {
                image::metadata_t *tmp_src;
                int chk;

                tmp_src = image::clone(src, 0, 16, 1);
                if (!tmp_src)
                    return -ENOMEM;

                chk = ctx->co_ctx->detect_corners(common::util::AVFrame2cvMat(tmp_src), cv::Size(win_width,win_height));
                image::free(&tmp_src);

                return chk;
            }

            int detect_corners(metadata_t *ctx, const image::metadata_t *src)
            {
                assert(src->format == AV_PIX_FMT_GRAY8 ||
                       src->format == AV_PIX_FMT_RGB24 ||
                       src->format == AV_PIX_FMT_BGR24 ||
                       src->format == AV_PIX_FMT_RGB32 ||
                       src->format == AV_PIX_FMT_BGR32);

                return _detect_corners(ctx, src);
            }

            double estimate_parameters(metadata_t *ctx)
            {
                return ctx->co_ctx->estimate_parameters();
            }

            int build_parameters(metadata_t *ctx, double alpha)
            {
                int chk = ctx->co_ctx->build_parameters(alpha);
                if (chk != 0)
                    return chk;

                image::free(&ctx->warper_xmap);
                image::free(&ctx->warper_ymap);

                ctx->warper_xmap = image::clone(ABCDK_XPU_PIXFMT_GRAYF32, ctx->co_ctx->m_warper_xmap,1, 16 ,0);
                if (!ctx->warper_xmap)
                    return -ENOMEM;

                ctx->warper_ymap = image::clone(ABCDK_XPU_PIXFMT_GRAYF32, ctx->co_ctx->m_warper_ymap,1, 16 ,0);
                if (!ctx->warper_ymap)
                    return -ENOMEM;

                return 0;
            }

            static abcdk_object_t *_dump_parameters(metadata_t *ctx, const char *magic)
            {
                std::string tmp_dst;
                int chk;

                chk = ctx->co_ctx->dump_parameters(tmp_dst, magic);
                if (chk != 0)
                    return NULL;

                return abcdk_object_copyfrom(tmp_dst.c_str(), tmp_dst.length());
            }

            abcdk_object_t *dump_parameters(metadata_t *ctx, const char *magic)
            {
                return _dump_parameters(ctx, magic);
            }

            int load_parameters(metadata_t *ctx, const char *src, const char *magic)
            {
                return ctx->co_ctx->load_parameters(src, magic);
            }

            static int _undistort(metadata_t *ctx, const image::metadata_t *src, image::metadata_t **dst, abcdk_xpu_inter_t inter_mode)
            {
                int chk;

                chk = image::reset(dst, src->width, src->height, pixfmt::ffmpeg_to_local(src->format), 16, 0);
                if (chk != 0)
                    return chk;

                chk = imgproc::remap(src, *dst, ctx->warper_xmap, ctx->warper_ymap, inter_mode);
                if (chk != 0)
                    return chk;

                return 0;
            }

            int undistort(metadata_t *ctx, const image::metadata_t *src, image::metadata_t **dst, abcdk_xpu_inter_t inter_mode)
            {
                assert(src->format == AV_PIX_FMT_GRAY8 ||
                       src->format == AV_PIX_FMT_RGB24 ||
                       src->format == AV_PIX_FMT_BGR24 ||
                       src->format == AV_PIX_FMT_RGB32 ||
                       src->format == AV_PIX_FMT_BGR32);

                return _undistort(ctx, src, dst, inter_mode);
            }
        } // namespace calibrate
    } // namespace nvidia
} // namespace abcdk_xpu
