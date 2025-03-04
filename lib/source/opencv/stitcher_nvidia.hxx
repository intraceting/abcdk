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
            std::vector<abcdk_torch_image_t *> m_cuda_warper_xmaps;
            std::vector<abcdk_torch_image_t *> m_cuda_warper_ymaps;

        public:
            stitcher_cuda()
            {
            }
            virtual ~stitcher_cuda()
            {
            }
        };
    } // namespace opencv
} // namespace abcdk

#endif // ABCDK_OPENCV_STITCHER_CUDA_HXX