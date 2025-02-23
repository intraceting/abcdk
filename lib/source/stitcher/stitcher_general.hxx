/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_STITCHER_GENERAL_HXX
#define ABCDK_STITCHER_GENERAL_HXX

#include "stitcher.hxx"

#ifdef OPENCV_CORE_HPP

namespace abcdk
{
    namespace stitcher
    {
        class stitcher_general : public stitcher
        {
        private:
            std::vector<mat> m_remap_outs;

        public:
            stitcher_general()
            {
            }

            virtual ~stitcher_general()
            {
            }

        protected:
            virtual bool remap(const std::vector<mat> &imgs)
            {
                assert(imgs.size() > 0);
                assert(imgs.size() >= m_img_good_idxs.size());
                assert(m_img_good_sizes.size() == m_img_good_idxs.size());
                assert(m_warper_rects.size() == m_img_good_idxs.size());
                assert(m_warper_xmaps.size() == m_img_good_idxs.size());
                assert(m_warper_ymaps.size() == m_img_good_idxs.size());

                /*输出的数量和顺序相同，但未能拼接的输出图像为空。*/
                if (m_remap_outs.size() != imgs.size())
                    m_remap_outs.resize(imgs.size());

                for (int i = 0; i < m_img_good_idxs.size(); i++)
                {
                    int idx = m_img_good_idxs[i];
                    int img_w = m_img_good_sizes[i].width;
                    int img_h = m_img_good_sizes[i].height;
                    int warper_w = m_warper_rects[i].width;
                    int warper_h = m_warper_rects[i].height;
                    auto &warper_xmap = m_warper_xmaps[i];
                    auto &warper_ymap = m_warper_ymaps[i];
                    auto &img = imgs[idx];
                    auto &remap_out = m_remap_outs[idx];

                    assert(img.type() == CV_8UC1 || img.type() == CV_8UC3 || img.type() == CV_8UC4);
                    assert(img.cols() == img_w && img.rows() == img_h);

                    /*创建变型后的图像存储空间。*/
                    remap_out.create(warper_h, warper_w, img.type());

                    if (remap_out.empty())
                        return false;

                    cv::remap(img.m_cuda_ctx, remap_out.m_host_ctx, warper_xmap, warper_ymap, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                }

                return true;
            }

            virtual bool compose(mat &out, bool optimize_seam = true)
            {
                int stuff[4] = {0};

                 /*创建全景图像存储空间。*/
                out.m_host_ctx.create(m_blend_height, m_blend_width, m_remap_outs[0].type());
                if (out.m_host_ctx.empty())
                    return false;

                for (int i = 0; i < m_blend_idxs.size(); i++)
                {
                    int idx = m_blend_idxs[i];
                    cv::Rect r = m_blend_rects[i];
                    auto &img = m_remap_outs[idx];

                    assert(img.type() == out.type());

                    /*计算重叠宽度。*/
                    int overlap_w = (i <= 0 ? 0 : (m_blend_rects[i - 1].width + m_blend_rects[i - 1].x - m_blend_rects[i].x));

                    abcdk::generic::imageproc::compose(out.m_host_ctx.channels(), true,
                                                       out.m_host_ctx.data, out.m_host_ctx.cols, out.m_host_ctx.step, out.m_host_ctx.rows,
                                                       img.data, img.cols, img.step, img.rows,
                                                       stuff, r.x, r.y, overlap_w, optimize_seam);
                }

                return true;
            }
        };
    } //    namespace stitcher
} // namespace abcdk

#endif // OPENCV_CORE_HPP

#endif // ABCDK_STITCHER_GENERAL_HXX