/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_STITCHER_CUDA_HXX
#define ABCDK_STITCHER_CUDA_HXX

#include "abcdk/cuda/avutil.h"
#include "stitcher_general.hxx"

#ifdef AVUTIL_AVUTIL_H
#ifdef OPENCV_CORE_HPP

namespace abcdk
{
    namespace stitcher
    {
        class stitcher_cuda : public abcdk::stitcher::stitcher_general
        {
        public:
            stitcher_cuda()
            {
            }

            virtual ~stitcher_cuda()
            {
            }

        private:
            std::vector<image> m_gpu_warper_xmaps;
            std::vector<image> m_gpu_warper_ymaps;

        protected:
            virtual void remap(std::vector<image> &outs, const std::vector<image> &imgs)
            {
                assert(imgs.size() > 0);
                assert(imgs.size() >= m_img_good_idxs.size());
                assert(m_img_good_sizes.size() == m_img_good_idxs.size());
                assert(m_warper_rects.size() == m_img_good_idxs.size());
                assert(m_warper_xmaps.size() == m_img_good_idxs.size());
                assert(m_warper_ymaps.size() == m_img_good_idxs.size());

                /*输出的数量和顺序相同，但未能拼接的输出图像为空。*/
                if (outs.size() != imgs.size())
                    outs.resize(imgs.size());

                /*可能还未复制，且仅复制一次即可。*/
                if (m_gpu_warper_xmaps.size() != m_img_good_sizes.size() || m_gpu_warper_ymaps.size() != m_img_good_sizes.size())
                {
                    m_gpu_warper_xmaps.resize(m_img_good_sizes.size());
                    m_gpu_warper_ymaps.resize(m_img_good_sizes.size());

                    for (int i = 0; i < m_img_good_sizes.size(); i++)
                    {
                        m_gpu_warper_xmaps[i].copyform(m_warper_xmaps[i]);
                        m_gpu_warper_ymaps[i].copyform(m_warper_ymaps[i]);
                    }
                }

                for (int i = 0; i < m_img_good_idxs.size(); i++)
                {
                    int idx = m_img_good_idxs[i];
                    int img_w = m_img_good_sizes[i].width;
                    int img_h = m_img_good_sizes[i].height;
                    int warper_w = m_warper_rects[i].width;
                    int warper_h = m_warper_rects[i].height;
                    auto &warper_xmap = m_gpu_warper_xmaps[i];
                    auto &warper_ymap = m_gpu_warper_ymaps[i];

                    assert(imgs[idx].width() == img_w && imgs[idx].height() == img_h);

                    outs[idx].create(warper_w, warper_h, imgs[idx].pixfmt());

                    abcdk_cuda_avframe_remap(outs[idx], NULL, imgs[idx], NULL, warper_xmap, warper_ymap, NPPI_INTER_CUBIC);
                }
            }

            virtual void compose_panorama(image &out, const std::vector<image> &imgs, int compose_kind = 1)
            {
                int stuff[4] = {0};

                /*如果画布不能复用，则创建新的画布。*/
                out.create(m_blend_width, m_blend_height, imgs[0].pixfmt());

                for (int i = 0; i < m_blend_idxs.size(); i++)
                {
                    int idx = m_blend_idxs[i];
                    cv::Rect r = m_blend_rects[i];

                    assert(imgs[idx].pixfmt() == out.pixfmt());

                    /*计算重叠宽度。*/
                    int overlap_w = (i <= 0 ? 0 : (m_blend_rects[i - 1].width + m_blend_rects[i - 1].x - m_blend_rects[i].x));

                    abcdk_cuda_avframe_compose(out, imgs[idx], stuff, r.x, r.y, overlap_w, ((compose_kind == 1) ? 1 : 0));
                }
            }
        };
    } //    namespace stitcher
} // namespace abcdk

#endif // #ifdef OPENCV_CORE_HPP
#endif // AVUTIL_AVUTIL_H

#endif // ABCDK_STITCHER_CUDA_HXX