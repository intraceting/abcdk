/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_CUDA_STITCHER_HXX
#define ABCDK_TORCH_CUDA_STITCHER_HXX

#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/opencv.h"
#include "../torch/stitcher.hxx"

#ifdef __cuda_cuda_h__

#ifdef OPENCV_STITCHING_STITCHER_HPP

namespace abcdk
{
    namespace torch_cuda
    {
        class stitcher : public torch::stitcher
        {
        private:
            CUcontext m_cuda_ctx;

            std::vector<abcdk_torch_image_t *> m_owner_warper_xmaps;
            std::vector<abcdk_torch_image_t *> m_owner_warper_ymaps;
            std::vector<abcdk_torch_image_t *> m_warper_outs;

        public:
            stitcher(CUcontext cuda_ctx)
            {
                m_cuda_ctx = cuda_ctx;
            }

            virtual ~stitcher()
            {
                ctx_push_current();

                for (auto &t : m_owner_warper_xmaps)
                    abcdk_torch_image_free_cuda(&t);

                for (auto &t : m_owner_warper_ymaps)
                    abcdk_torch_image_free_cuda(&t);

                for (auto &t : m_warper_outs)
                    abcdk_torch_image_free_cuda(&t);

                ctx_pop_current();
            }

        protected:
            virtual void ctx_push_current()
            {
                cuCtxPushCurrent(m_cuda_ctx);
            }

            virtual void ctx_pop_current()
            {
                cuCtxPopCurrent(NULL);
            }

            virtual bool remap(const std::vector<abcdk_torch_image_t *> &imgs)
            {
                int chk;

                assert(imgs.size() > 0);
                assert(imgs.size() >= m_img_good_idxs.size());
                assert(m_img_good_sizes.size() == m_img_good_idxs.size());
                assert(m_warper_rects.size() == m_img_good_idxs.size());
                assert(m_warper_xmaps.size() == m_img_good_idxs.size());
                assert(m_warper_ymaps.size() == m_img_good_idxs.size());

                /*输出的数量和顺序相同，但未能拼接的输出图像为空。*/
                if (m_warper_outs.size() != imgs.size())
                {
                    for (auto &t : m_warper_outs)
                        abcdk_torch_image_free_cuda(&t);

                    for (int i = 0; i < imgs.size(); i++)
                        m_warper_outs.push_back(abcdk_torch_image_alloc_cuda());
                }

                /*可能还未复制，且仅复制一次即可。*/
                if (m_owner_warper_xmaps.size() != m_img_good_sizes.size() ||
                    m_owner_warper_ymaps.size() != m_img_good_sizes.size())
                {
                    for (auto &t : m_owner_warper_xmaps)
                        abcdk_torch_image_free_cuda(&t);

                    for (auto &t : m_owner_warper_ymaps)
                        abcdk_torch_image_free_cuda(&t);

                    m_owner_warper_xmaps.resize(m_img_good_sizes.size());
                    m_owner_warper_ymaps.resize(m_img_good_sizes.size());

                    for (int i = 0; i < m_img_good_sizes.size(); i++)
                    {
                        m_owner_warper_xmaps[i] = abcdk_torch_image_create_cuda(m_warper_xmaps[i].cols, m_warper_xmaps[i].rows, ABCDK_TORCH_PIXFMT_GRAYF32, 1);
                        if (!m_owner_warper_xmaps[i])
                            return false;

                        abcdk_torch_image_copy_plane_cuda(m_owner_warper_xmaps[i], 0, m_warper_xmaps[i].data, m_warper_xmaps[i].step);

                        m_owner_warper_ymaps[i] = abcdk_torch_image_create_cuda(m_warper_ymaps[i].cols, m_warper_ymaps[i].rows, ABCDK_TORCH_PIXFMT_GRAYF32, 1);
                        if (!m_owner_warper_ymaps[i])
                            return false;

                        abcdk_torch_image_copy_plane_cuda(m_owner_warper_ymaps[i], 0, m_warper_ymaps[i].data, m_warper_ymaps[i].step);
                    }
                }

                for (int i = 0; i < m_img_good_idxs.size(); i++)
                {
                    int idx = m_img_good_idxs[i];
                    int img_w = m_img_good_sizes[i].width;
                    int img_h = m_img_good_sizes[i].height;
                    int warper_w = m_warper_rects[i].width;
                    int warper_h = m_warper_rects[i].height;
                    abcdk_torch_image_t *xmap_it = m_owner_warper_xmaps[i];
                    abcdk_torch_image_t *ymap_it = m_owner_warper_ymaps[i];
                    abcdk_torch_image_t *outs_it = m_warper_outs[idx];
                    abcdk_torch_image_t *imgs_it = imgs[idx];

                    assert(imgs_it->width == img_w && imgs_it->height == img_h);

                    /*创建变换后的图像存储空间。*/
                    chk = abcdk_torch_image_reset_cuda(&outs_it, warper_w, warper_h, imgs_it->pixfmt, 1);
                    if (chk != 0)
                        return false;

                    abcdk_torch_imgproc_remap_cuda(outs_it, NULL, imgs_it, NULL, xmap_it, ymap_it, NPPI_INTER_CUBIC);
                }

                return true;
            }

            virtual bool compose(abcdk_torch_image_t *out, bool optimize_seam = true)
            {
                uint32_t scalar[4] = {0};
                int chk;

                assert(m_warper_outs.size() >= 0);

                /*创建全景图像存储空间。*/
                chk = abcdk_torch_image_reset_cuda(&out, m_blend_width, m_blend_height, m_warper_outs[m_img_good_idxs[0]]->pixfmt, 1);
                if (chk != 0)
                    return false;

                for (int i = 0; i < m_blend_idxs.size(); i++)
                {
                    int idx = m_blend_idxs[i];
                    cv::Rect r = m_blend_rects[i];
                    abcdk_torch_image_t *imgs_it = m_warper_outs[idx];

                    assert(imgs_it->pixfmt == out->pixfmt);

                    /*计算重叠宽度。*/
                    int overlap_w = (i <= 0 ? 0 : (m_blend_rects[i - 1].width + m_blend_rects[i - 1].x - m_blend_rects[i].x));

                    abcdk_torch_imgproc_compose_cuda(out, imgs_it, scalar, r.x, r.y, overlap_w, (optimize_seam ? 1 : 0));
                }

                return true;
            }
        };

    } // namespace torch_cuda
} // namespace abcdk

#endif // OPENCV_STITCHING_STITCHER_HPP

#endif //__cuda_cuda_h__

#endif // ABCDK_TORCH_CUDA_STITCHER_HXX