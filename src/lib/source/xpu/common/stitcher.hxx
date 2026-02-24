/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_STITCHER_HXX
#define ABCDK_XPU_COMMON_STITCHER_HXX

#include "abcdk/xpu/stitcher.h"
#include "../runtime.in.h"

namespace abcdk_xpu
{
    namespace common
    {
        class stitcher
        {
        public:
            std::vector<int> m_img_good_idxs;
            std::vector<cv::Size> m_img_good_sizes;
            std::vector<cv::detail::CameraParams> m_camera_params;
            std::vector<cv::Mat> m_warper_xmaps;
            std::vector<cv::Mat> m_warper_ymaps;
            std::vector<cv::Rect> m_warper_rects;
            std::vector<int> m_blend_idxs;
            std::vector<cv::Rect> m_blend_rects;
            cv::Size m_panorama_size;
        public:
            static std::shared_ptr<stitcher> create();
        protected:
            stitcher();
            virtual ~stitcher();
        public:

        public:
            virtual int set_feature_finder(const char *name) = 0;
            virtual int set_feature_matcher(const char *name, float threshold = 0.3) = 0;
            virtual int set_estimator(const char *name) = 0;
            virtual int set_bundle_adjuster(const char *name) = 0;
            virtual int set_warper(const char *name, float scale = 1.0) = 0;

            virtual int estimate_parameters(const std::vector<cv::Mat> &imgs, const std::vector<cv::Mat> &masks,
                                            float good_threshold = 0.8, float adjuster_threshold = 0.8) = 0;
                                            
            virtual int build_parameters() = 0;

            virtual int dump_parameters(std::string &dst, const char *magic = NULL) = 0;
            virtual int load_parameters(const char *src,  const char *magic = NULL) = 0;
        };
    } // namespace common
} // namespace abcdk_xpu

#endif //ABCDK_XPU_COMMON_STITCHER_HXX