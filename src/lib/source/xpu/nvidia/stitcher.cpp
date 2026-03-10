/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "stitcher.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace stitcher
        {
            typedef struct _metadata
            {
                std::shared_ptr<common::stitcher> co_ctx;
                std::vector<image::metadata_t *> warper_xmaps;
                std::vector<image::metadata_t *> warper_ymaps;
                std::vector<image::metadata_t *> warper_parts;
            } metadata_t;

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                for (auto &one : ctx_p->warper_xmaps)
                    image::free(&one);

                for (auto &one : ctx_p->warper_ymaps)
                    image::free(&one);

                for (auto &one : ctx_p->warper_parts)
                    image::free(&one);

                delete ctx_p;
            }

            metadata_t *alloc()
            {
                metadata_t *ctx;

                ctx = new metadata_t;
                if (!ctx)
                    return NULL;

                ctx->co_ctx = common::stitcher::create();

                return ctx;
            }

            int set_feature_finder(metadata_t *ctx, const char *name)
            {
                return ctx->co_ctx->set_feature_finder(name);
            }

            int set_warper(metadata_t *ctx, const char *name)
            {
                return ctx->co_ctx->set_warper(name);
            }

            static int _estimate_parameters(metadata_t *ctx, std::vector<image::metadata_t *> &img, std::vector<image::metadata_t *> &mask, float threshold)
            {
                std::vector<cv::Mat> tmp_img;
                std::vector<cv::Mat> tmp_mask;

                assert(img.size() == mask.size());

                tmp_img.resize(img.size());
                tmp_mask.resize(img.size());

                for (int i = 0; i < img.size(); i++)
                {
                    tmp_img[i] = common::util::AVFrame2cvMat(img[i]);

                    if (mask[i])
                        tmp_mask[i] = common::util::AVFrame2cvMat(mask[i]);
                    else
                        tmp_mask[i] = cv::Mat(img[i]->height, img[i]->width, CV_8UC1, cv::Scalar(255, 255, 255)); // white.
                }

                return ctx->co_ctx->estimate_parameters(tmp_img, tmp_mask, threshold, threshold);//匹配和筛选阈值设置为相同的.
            }

            static int _estimate_parameters(metadata_t *ctx, int count, const image::metadata_t *img[], const image::metadata_t *mask[], float threshold)
            {
                std::vector<image::metadata_t *> tmp_img(count);
                std::vector<image::metadata_t *> tmp_mask(count);
                int chk;

                for (int i = 0; i < count; i++)
                {
                    tmp_img[i] = image::clone(img[i], 0, 16, 1);

                    if (mask && mask[i])
                        tmp_mask[i] = image::clone(mask[i], 0, 16, 1);
                }

                chk = _estimate_parameters(ctx, tmp_img, tmp_mask, threshold);

                for (int i = 0; i < count; i++)
                {
                    image::free(&tmp_img[i]);
                    image::free(&tmp_mask[i]);
                }

                return chk;
            }

            int estimate_parameters(metadata_t *ctx, int count, const image::metadata_t *img[], const image::metadata_t *mask[], float threshold)
            {
                for (int i = 0; i < count; i++)
                {
                    auto &img_p = img[i];

                    assert(img_p != NULL);
                    assert(img_p->format == AV_PIX_FMT_GRAY8 ||
                           img_p->format == AV_PIX_FMT_RGB24 ||
                           img_p->format == AV_PIX_FMT_BGR24 ||
                           img_p->format == AV_PIX_FMT_RGB32 ||
                           img_p->format == AV_PIX_FMT_BGR32);

                    assert(img[i]->format == img[(i + 1) % count]->format);
                    // assert(img[i]->width == img[(i + 1) % count]->width);
                    // assert(img[i]->height == img[(i + 1) % count]->height);

                    if (!mask)
                        continue;

                    auto &mask_p = mask[i];

                    assert(img_p->width == mask_p->width);
                    assert(img_p->height == mask_p->height);

                    assert(mask_p->format == AV_PIX_FMT_GRAY8 ||
                           mask_p->format == AV_PIX_FMT_RGB24 ||
                           mask_p->format == AV_PIX_FMT_BGR24 ||
                           mask_p->format == AV_PIX_FMT_RGB32 ||
                           mask_p->format == AV_PIX_FMT_BGR32);

                    // assert(mask[i]->width == mask[(i + 1) % count]->width);
                    // assert(mask[i]->height == mask[(i + 1) % count]->height);
                }

                return _estimate_parameters(ctx, count, img, mask, threshold);
            }

            int build_parameters(metadata_t *ctx)
            {
                int chk = ctx->co_ctx->build_parameters();
                if (chk != 0)
                    return chk;

                for (auto &one : ctx->warper_xmaps)
                    image::free(&one);

                for (auto &one : ctx->warper_ymaps)
                    image::free(&one);

                for (auto &one : ctx->warper_parts)
                    image::free(&one);

                ctx->warper_xmaps.resize(ctx->co_ctx->m_img_good_sizes.size());
                ctx->warper_ymaps.resize(ctx->co_ctx->m_img_good_sizes.size());

                for (int i = 0; i < ctx->co_ctx->m_img_good_sizes.size(); i++)
                {
                    ctx->warper_xmaps[i] = image::clone(ABCDK_XPU_PIXFMT_GRAYF32, ctx->co_ctx->m_warper_xmaps[i], 1, 16, 0);
                    if (!ctx->warper_xmaps[i])
                        return -ENOMEM;

                    ctx->warper_ymaps[i] = image::clone(ABCDK_XPU_PIXFMT_GRAYF32, ctx->co_ctx->m_warper_ymaps[i], 1, 16, 0);
                    if (!ctx->warper_ymaps[i])
                        return -ENOMEM;
                }

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

            static int _compose(metadata_t *ctx, int count, const image::metadata_t *img[], image::metadata_t **out, int optimize_seam)
            {
                abcdk_xpu_scalar_t scalar = {0};
                int chk;

                if (ctx->warper_parts.size() <= 0)
                    ctx->warper_parts.resize(count);

                assert(count == ctx->warper_parts.size());

                for (int i = 0; i < ctx->co_ctx->m_img_good_idxs.size(); i++)
                {
                    int idx = ctx->co_ctx->m_img_good_idxs[i];
                    int img_w = ctx->co_ctx->m_img_good_sizes[i].width;
                    int img_h = ctx->co_ctx->m_img_good_sizes[i].height;
                    int warper_w = ctx->co_ctx->m_warper_rects[i].width;
                    int warper_h = ctx->co_ctx->m_warper_rects[i].height;
                    auto &xmap_p = ctx->warper_xmaps[i];
                    auto &ymap_p = ctx->warper_ymaps[i];
                    auto &img_p = img[idx];

                    assert(img_p->width == img_w && img_p->height == img_h);

                    chk = image::reset(&ctx->warper_parts[idx], warper_w, warper_h, pixfmt::ffmpeg_to_local(img_p->format), 16, 0);
                    if (chk != 0)
                        return chk;

                    imgproc::remap(img_p, ctx->warper_parts[idx], xmap_p, ymap_p, ABCDK_XPU_INTER_CUBIC);
                }

                chk = image::reset(out, ctx->co_ctx->m_panorama_size.width, ctx->co_ctx->m_panorama_size.height,
                                   pixfmt::ffmpeg_to_local(ctx->warper_parts[ctx->co_ctx->m_img_good_idxs[0]]->format), 16, 0);

                if (chk != 0)
                    return chk;

                image::zero(*out);//set 0.

                for (int i = 0; i < ctx->co_ctx->m_blend_idxs.size(); i++)
                {
                    int idx = ctx->co_ctx->m_blend_idxs[i];
                    cv::Rect r = ctx->co_ctx->m_blend_rects[i];
                    auto part_p = ctx->warper_parts[idx];

                    assert(part_p->format == (*out)->format);

                    /*计算重叠宽度.*/
                    int overlap_w = (i <= 0 ? 0 : (ctx->co_ctx->m_blend_rects[i - 1].width + ctx->co_ctx->m_blend_rects[i - 1].x - ctx->co_ctx->m_blend_rects[i].x));

                    imgproc::compose(*out, part_p, r.x, r.y, overlap_w, &scalar, optimize_seam);
                }

                return 0;
            }

            int compose(metadata_t *ctx, int count, const image::metadata_t *img[], image::metadata_t **out, int optimize_seam)
            {
                for (int i = 0; i < count; i++)
                {
                    auto &img_p = img[i];

                    assert(img_p != NULL);
                    assert(img_p->format == AV_PIX_FMT_GRAY8 ||
                           img_p->format == AV_PIX_FMT_RGB24 ||
                           img_p->format == AV_PIX_FMT_BGR24 ||
                           img_p->format == AV_PIX_FMT_RGB32 ||
                           img_p->format == AV_PIX_FMT_BGR32);

                    assert(img[i]->format == img[(i + 1) % count]->format);
                }

                return _compose(ctx, count, img, out, optimize_seam);
            }

        } // namespace stitcher
    } // namespace nvidia

} // namespace abcdk_xpu
