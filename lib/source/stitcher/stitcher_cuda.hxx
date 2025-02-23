/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_STITCHER_CUDA_HXX
#define ABCDK_STITCHER_CUDA_HXX

#include "stitcher.hxx"

#ifdef __cuda_cuda_h__
#ifdef OPENCV_CORE_HPP

namespace abcdk
{
    namespace stitcher
    {
        class stitcher_cuda : public stitcher
        {
        private:
            std::vector<mat> m_cuda_warper_xmaps;
            std::vector<mat> m_cuda_warper_ymaps;
            std::vector<mat> m_cuda_remap_outs;

        public:
            stitcher_cuda()
            {
            }

            virtual ~stitcher_cuda()
            {
                for (auto &it : m_gpu_warper_xmaps)
                    abcdk_ndarray_free(&it);

                for (auto &it : m_gpu_warper_ymaps)
                    abcdk_ndarray_free(&it);
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
                if (m_cuda_remap_outs.size() != imgs.size())
                    m_cuda_remap_outs.resize(imgs.size());

                /*可能还未复制，且仅复制一次即可。*/
                if (m_gpu_warper_xmaps.size() != m_img_good_sizes.size() || m_gpu_warper_ymaps.size() != m_img_good_sizes.size())
                {
                    m_gpu_warper_xmaps.resize(m_img_good_sizes.size());
                    m_gpu_warper_ymaps.resize(m_img_good_sizes.size());

                    for (int i = 0; i < m_img_good_sizes.size(); i++)
                    {
                        m_gpu_warper_xmaps[i].clone(mat::memory_kind::CUDA,m_warper_xmaps[i]);
                        m_gpu_warper_ymaps[i].clone(mat::memory_kind::CUDA,m_warper_ymaps[i]);
                    }
                }

                for (int i = 0; i < m_img_good_idxs.size(); i++)
                {
                    int idx = m_img_good_idxs[i];
                    int img_w = m_img_good_sizes[i].width;
                    int img_h = m_img_good_sizes[i].height;
                    int warper_w = m_warper_rects[i].width;
                    int warper_h = m_warper_rects[i].height;
                    auto &warper_xmap = m_cuda_warper_xmaps[i];
                    auto &warper_ymap = m_cuda_warper_ymaps[i];
                    auto &img = imgs[idx];
                    auto &remap_out = m_remap_outs[idx];

                    assert(img.type(mat::memory_kind::CUDA) == CV_8UC1 || img.type(mat::memory_kind::CUDA) == CV_8UC3 || img.type(mat::memory_kind::CUDA) == CV_8UC4);
                    assert(img.cols(mat::memory_kind::CUDA) == img_w && img.rows(mat::memory_kind::CUDA) == img_h);

                    /*创建变型后的图像存储空间。*/
                    remap_out.create(warper_h, warper_w, img.type(mat::memory_kind::CUDA));

                    if (remap_out.empty(mat::memory_kind::CUDA))
                        return false;

                    //abcdk_cuda_avframe_remap(remap_out.m_cuda_ctx, NULL, img.m_cuda_ctx, NULL, warper_xmap, warper_ymap, NPPI_INTER_CUBIC);
                }

                return true;
            }

            virtual bool compose_panorama(mat &out, bool optimize_seam = true)
            {
                int stuff[4] = {0};

                /*创建全景图像存储空间。*/
                out.create(m_blend_width, m_blend_height, m_remap_outs[0].type(mat::memory_kind::CUDA),mat::memory_kind::CUDA);

                for (int i = 0; i < m_blend_idxs.size(); i++)
                {
                    int idx = m_blend_idxs[i];
                    cv::Rect r = m_blend_rects[i];
                    auto &img = m_remap_outs[idx];

                    assert(img.type(mat::memory_kind::CUDA) == out.type(mat::memory_kind::CUDA));

                    /*计算重叠宽度。*/
                    int overlap_w = (i <= 0 ? 0 : (m_blend_rects[i - 1].width + m_blend_rects[i - 1].x - m_blend_rects[i].x));

                    abcdk_cuda_imgproc_compose_8u_C1R(out.m_cuda_ctx->data, out.m_cuda_ctx->width, out.m_cuda_ctx->stride, out.m_cuda_ctx->height,
                                                      , img.m_cuda_ctx->data, img.m_cuda_ctx->width, img.m_cuda_ctx->stride, img.m_cuda_ctx->height,
                                                      stuff, r.x, r.y, overlap_w, optimize_seam);
                }
            }
        };
    } //    namespace stitcher
} // namespace abcdk

#endif // OPENCV_CORE_HPP
#endif // __cuda_cuda_h__

#endif // ABCDK_STITCHER_CUDA_HXX