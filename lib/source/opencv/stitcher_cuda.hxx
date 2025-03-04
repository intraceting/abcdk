/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_OPENCV_STITCHER_CUDA_HXX
#define ABCDK_OPENCV_STITCHER_CUDA_HXX

#include "stitcher.hxx"

#ifdef OPENCV_STITCHING_STITCHER_HPP

namespace abcdk
{
    namespace opencv
    {
        class stitcher_cuda : public stitcher
        {
        protected:
            std::vector<abcdk_torch_image_t *> m_owner_warper_xmaps;
            std::vector<abcdk_torch_image_t *> m_owner_warper_ymaps;
        public:
            stitcher_cuda()
            {
            }

            virtual ~stitcher_cuda()
            {
                for (auto &t : m_owner_warper_xmaps)
                    abcdk_torch_image_free(&t);

                for (auto &t : m_owner_warper_ymaps)
                    abcdk_torch_image_free(&t);
            }

        private:
            bool update_warper_maps()
            {
                /*可能还未复制，且仅复制一次即可。*/
                if (m_owner_warper_xmaps.size() != m_img_good_sizes.size() ||
                    m_owner_warper_ymaps.size() != m_img_good_sizes.size())
                {
                    for (auto &t : m_owner_warper_xmaps)
                        abcdk_torch_image_free(&t);

                    for (auto &t : m_owner_warper_ymaps)
                        abcdk_torch_image_free(&t);

                    m_owner_warper_xmaps.resize(m_img_good_sizes.size());
                    m_owner_warper_ymaps.resize(m_img_good_sizes.size());

                    for (int i = 0; i < m_img_good_sizes.size(); i++)
                    {
                        m_owner_warper_xmaps[i] = abcdk_cuda_image_create(m_warper_xmaps[i].cols, m_warper_xmaps[i].rows, ABCDK_TORCH_PIXFMT_GRAYF32, 1);

                        abcdk_cuda_image_copy_plane(m_owner_warper_xmaps[i], 0, m_warper_xmaps[i].data, m_warper_xmaps[i].step);

                        m_owner_warper_ymaps[i] = abcdk_cuda_image_create(m_warper_ymaps[i].cols, m_warper_ymaps[i].rows, ABCDK_TORCH_PIXFMT_GRAYF32, 1);

                        abcdk_cuda_image_copy_plane(m_owner_warper_ymaps[i], 0, m_warper_ymaps[i].data, m_warper_ymaps[i].step);
                    }
                }

                return true;
            }
            protected:
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
                        abcdk_torch_image_free(&t);

                    m_warper_outs.resize(imgs.size());
                }

                update_owner_warper_maps();

                for (int i = 0; i < m_img_good_idxs.size(); i++)
                {
                    int idx = m_img_good_idxs[i];
                    int img_w = m_img_good_sizes[i].width;
                    int img_h = m_img_good_sizes[i].height;
                    int warper_w = m_warper_rects[i].width;
                    int warper_h = m_warper_rects[i].height;
                    auto &xmap_it = m_owner_warper_xmaps[i];
                    auto &ymap_it = m_owner_warper_ymaps[i];
                    auto &outs_it = m_warper_outs[idx];
                    auto &imgs_it = imgs[idx];

                    assert(imgs_it->width == img_w && imgs_it->height == img_h);

                    /*创建变换后的图像存储空间。*/
                    chk = abcdk_cuda_image_reset(&outs_it, warper_h, warper_w, imgs_it->pixfmt, 1);
                    if (chk != 0)
                        return false;

                    abcdk_cuda_imgproc_remap_8u(outs_it, NULL, imgs_it, NULL, xmap_it, ymap_it, cv::INTER_CUBIC);
                }

                return true;
            }

            virtual bool compose(abcdk_torch_image_t *out, bool optimize_seam = true)
            {
                uint8_t scalar[4] = {0};
                int chk;

                assert(m_warper_outs.size() >= 0);

                /*创建全景图像存储空间。*/
                chk = abcdk_cuda_image_reset(&out, m_blend_height, m_blend_width, m_warper_outs[0]->pixfmt, 1);
                if (chk != 0)
                    return false;

                for (int i = 0; i < m_blend_idxs.size(); i++)
                {
                    int idx = m_blend_idxs[i];
                    cv::Rect r = m_blend_rects[i];
                    auto &imgs_it = m_warper_outs[idx];

                    assert(imgs_it->pixfmt == out->pixfmt);

                    /*计算重叠宽度。*/
                    int overlap_w = (i <= 0 ? 0 : (m_blend_rects[i - 1].width + m_blend_rects[i - 1].x - m_blend_rects[i].x));

                    abcdk_cuda_imgproc_compose_8u(out, imgs_it, scalar, r.x, r.y, overlap_w, (optimize_seam ? 1 : 0));
                }

                return true;
            }
        };
    } // namespace opencv
} // namespace abcdk

#endif // OPENCV_STITCHING_STITCHER_HPP

#endif // ABCDK_OPENCV_STITCHER_CUDA_HXX