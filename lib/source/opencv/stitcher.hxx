/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_OPENCV_STITCHER_HXX
#define ABCDK_OPENCV_STITCHER_HXX

#include "abcdk/opencv/opencv.h"
#include "abcdk/media/image.h"
#include "../generic/imageproc.hxx"

#ifdef OPENCV_STITCHING_STITCHER_HPP

namespace abcdk
{
    namespace opencv
    {
        class stitcher
        {
        public:
            static int Dump(std::string &metadata, stitcher &obj, const char *magic = NULL)
            {
                cv::FileStorage f("{}", cv::FileStorage::MEMORY | cv::FileStorage::WRITE | cv::FileStorage::FORMAT_XML);
                if (!f.isOpened())
                    return -1;

                if (magic && *magic)
                    cv::write(f, "magic", magic);

                cv::write(f, "good_count", (int)obj.m_img_good_idxs.size());

                for (int i = 0; i < obj.m_img_good_idxs.size(); i++)
                {
                    std::string key = "good_idxs_";
                    key += std::to_string(i);

                    cv::write(f, key, obj.m_img_good_idxs[i]);
                }

                for (int i = 0; i < obj.m_img_good_sizes.size(); i++)
                {
                    std::string key = "good_sizes_";
                    key += std::to_string(i);

                    cv::write(f, key, obj.m_img_good_sizes[i]);
                }

                for (int i = 0; i < obj.m_camera_params.size(); i++)
                {
                    std::string key = "camera_params_";
                    key += std::to_string(i);
                    key += "_focal";
                    cv::write(f, key, obj.m_camera_params[i].focal);

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_aspect";
                    cv::write(f, key, obj.m_camera_params[i].aspect);

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_ppx";
                    cv::write(f, key, obj.m_camera_params[i].ppx);

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_ppy";
                    cv::write(f, key, obj.m_camera_params[i].ppy);

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_R";
                    cv::write(f, key, obj.m_camera_params[i].R);

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_t";
                    cv::write(f, key, obj.m_camera_params[i].t);
                }

                metadata = f.releaseAndGetString();

                return 0;
            }

            static int Load(const char *metadata, stitcher &obj, const char *magic = NULL)
            {
                std::string old_magic;

                assert(metadata != NULL);

                cv::FileStorage f(metadata, cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_XML);
                if (!f.isOpened())
                    return -1;

                cv::FileNode node = f["magic"];
                if (!node.empty())
                {
                    old_magic = node.string();
                    if (old_magic.compare(magic))
                        return -127;
                }
                else if (magic && *magic)
                {
                    return -127;
                }

                node = f["good_count"];
                if (node.empty())
                    return -2;

                int good_count = node;
                if (good_count <= 0)
                    return -2;

                obj.m_img_good_idxs.resize(good_count);
                obj.m_img_good_sizes.resize(good_count);
                obj.m_camera_params.resize(good_count);

                for (int i = 0; i < obj.m_img_good_idxs.size(); i++)
                {
                    std::string key = "good_idxs_";
                    key += std::to_string(i);

                    node = f[key.c_str()];
                    obj.m_img_good_idxs[i] = node;
                }

                for (int i = 0; i < obj.m_img_good_sizes.size(); i++)
                {
                    std::string key = "good_sizes_";
                    key += std::to_string(i);

                    node = f[key.c_str()];
                    obj.m_img_good_sizes[i] = cv::Size(node[0], node[1]);
                }

                for (int i = 0; i < obj.m_camera_params.size(); i++)
                {
                    std::string key = "camera_params_";
                    key += std::to_string(i);
                    key += "_focal";

                    node = f[key.c_str()];
                    obj.m_camera_params[i].focal = node;

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_aspect";

                    node = f[key.c_str()];
                    obj.m_camera_params[i].aspect = node;

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_ppx";

                    node = f[key.c_str()];
                    obj.m_camera_params[i].ppx = node;

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_ppy";

                    node = f[key.c_str()];
                    obj.m_camera_params[i].ppy = node;

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_R";

                    node = f[key.c_str()];
                    obj.m_camera_params[i].R = node.mat();

                    key = "camera_params_";
                    key += std::to_string(i);
                    key += "_t";

                    node = f[key.c_str()];
                    obj.m_camera_params[i].t = node.mat();
                }

                obj.m_panorama_param_ok = false; // Not OK.
                obj.m_camera_param_ok = true;    // OK.

                return 0;
            }

        protected:
            cv::Ptr<cv::Feature2D> m_feature_finder;
            cv::Ptr<cv::detail::FeaturesMatcher> m_feature_matcher;
            cv::Ptr<cv::detail::Estimator> m_estimator;
            cv::Ptr<cv::detail::BundleAdjusterBase> m_bundle_adjuster;
            cv::Ptr<cv::detail::RotationWarper> m_rotation_warper;

        protected:
            std::vector<cv::detail::ImageFeatures> m_img_features;
            std::vector<cv::detail::MatchesInfo> m_img_matches;
            std::vector<int> m_img_good_idxs;
            std::vector<cv::Size> m_img_good_sizes;
            std::vector<cv::detail::CameraParams> m_camera_params;
            std::vector<cv::Mat> m_warper_xmaps;
            std::vector<cv::Mat> m_warper_ymaps;
            std::vector<cv::Rect> m_warper_rects;
            std::vector<cv::Rect> m_screen_rects;
            int m_blend_width;
            int m_blend_height;
            std::vector<int> m_blend_idxs;
            std::vector<cv::Rect> m_blend_rects;
            bool m_camera_param_ok;
            bool m_panorama_param_ok;

        public:
            stitcher()
            {
                m_camera_param_ok = false;
                m_panorama_param_ok = false;
            }

            virtual ~stitcher()
            {
            }

        protected:
            void find_feature(const std::vector<cv::Mat> &imgs, const std::vector<cv::Mat> &masks)
            {
                assert(imgs.size() > 0);
                assert(masks.size() == 0 || imgs.size() == masks.size());

                if (m_feature_finder.get() == NULL)
                    set_feature_finder("SIFT");

                m_img_features.resize(imgs.size());

                for (int i = 0; i < imgs.size(); i++)
                {
                    cv::Mat gray, mask;

                    mask = (masks.size() > 0 ? masks[i] : cv::Mat());

                    if (imgs[i].channels() == 3)
                        cv::cvtColor(imgs[i], gray, cv::COLOR_RGB2GRAY);
                    else if (imgs[i].channels() == 4)
                        cv::cvtColor(imgs[i], gray, cv::COLOR_RGBA2GRAY);
                    else
                        gray = imgs[i];

                    m_img_features[i].img_idx = i;
                    m_img_features[i].img_size = imgs[i].size();

                    m_feature_finder->detectAndCompute(gray, mask, m_img_features[i].keypoints, m_img_features[i].descriptors);

                    if (getenv("ABCDK_OPENCV_STITCHER_FIND_FEATURE_DUMP"))
                    {
                        cv::Mat img_tmp;

                        cv::drawKeypoints(imgs[i], m_img_features[i].keypoints, img_tmp, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        std::string img_name = "./abcdk-opencv-stitcher-find-feature-image-";
                        std::string mask_name = "./abcdk-opencv-stitcher-find-feature-mask-";
                        img_name += std::to_string(i);
                        mask_name += std::to_string(i);
                        img_name += ".jpg";
                        mask_name += ".jpg";

                        cv::imwrite(img_name, img_tmp);

                        if (!mask.empty())
                            cv::imwrite(mask_name, mask);
                    }
                }
            }

            bool match_feature()
            {
                if (m_img_features.size() <= 0)
                    return false;

                if (m_feature_matcher.get() == NULL)
                    set_feature_matcher("Best");

                (*m_feature_matcher)(m_img_features, m_img_matches);
                m_feature_matcher->collectGarbage();

                if (m_img_matches.size() <= 0)
                    return false;

                return true;
            }

            bool leave_biggest_component(const std::vector<cv::Mat> &imgs, float threshold = 0.8)
            {
                assert(imgs.size() > 0);
                assert(imgs.size() == m_img_features.size());

                m_img_good_idxs = cv::detail::leaveBiggestComponent(m_img_features, m_img_matches, threshold);

                m_img_good_sizes.resize(m_img_good_idxs.size());
                for (int i = 0; i < m_img_good_idxs.size(); i++)
                    m_img_good_sizes[i] = imgs[m_img_good_idxs[i]].size();

                if (m_img_good_sizes.size() <= 1)
                    return false;

                return true;
            }

            void draw_keypoints_matches(const std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &outs)
            {
                assert(m_img_good_idxs.size() <= imgs.size());
                assert(m_img_features.size() == m_img_good_idxs.size());
                assert(m_img_matches.size() >= m_img_good_idxs.size());

                outs.resize(m_img_matches.size());
                for (int i = 0; i < m_img_matches.size(); i++)
                {
                    if (m_img_matches[i].confidence <= 0.0)
                        continue;

                    int dst_idx = m_img_matches[i].dst_img_idx;
                    int src_idx = m_img_matches[i].src_img_idx;

                    if (src_idx == -1 && dst_idx != -1)
                    {
                        int dst_img = m_img_good_idxs[dst_idx];
                        // int src_img = m_img_good_idxs[src_idx];

                        cv::drawKeypoints(imgs[dst_img], m_img_features[dst_idx].keypoints, outs[i], cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    }
                    else if (src_idx != -1 && dst_idx == -1)
                    {
                        // int dst_img = m_img_good_idxs[dst_idx];
                        int src_img = m_img_good_idxs[src_idx];

                        cv::drawKeypoints(imgs[src_img], m_img_features[src_idx].keypoints, outs[i], cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    }
                    else if (src_idx != -1 && dst_idx != -1)
                    {
                        int dst_img = m_img_good_idxs[dst_idx];
                        int src_img = m_img_good_idxs[src_idx];

                        cv::drawMatches(imgs[src_img], m_img_features[src_idx].keypoints, imgs[dst_img], m_img_features[dst_idx].keypoints,
                                        m_img_matches[i].matches, outs[i], cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    }
                }
            }

            bool estimate_camera()
            {
                if (m_estimator.get() == NULL)
                    set_estimator("Homography");

                if (!(*m_estimator)(m_img_features, m_img_matches, m_camera_params))
                    return false;

                for (int i = 0; i < m_camera_params.size(); i++)
                {
                    cv::Mat R;
                    m_camera_params[i].R.convertTo(R, CV_32F);
                    m_camera_params[i].R = R;
                }

                return true;
            }

            bool camera_adjuster(float threshold = 0.8, const char mask[5] = "xxxxx")
            {
                if (m_bundle_adjuster.get() == NULL)
                    set_bundle_adjuster("ray");

                m_bundle_adjuster->setConfThresh(threshold);

                cv::Mat1b refine_mask = cv::Mat::zeros(3, 3, CV_8U);
                if (mask[0] == 'x')
                    refine_mask(0, 0) = 1;
                if (mask[1] == 'x')
                    refine_mask(0, 1) = 1;
                if (mask[2] == 'x')
                    refine_mask(0, 2) = 1;
                if (mask[3] == 'x')
                    refine_mask(1, 1) = 1;
                if (mask[4] == 'x')
                    refine_mask(1, 2) = 1;

                m_bundle_adjuster->setRefinementMask(refine_mask);
                if (!(*m_bundle_adjuster)(m_img_features, m_img_matches, m_camera_params))
                    return false;

                return true;
            }

            void camera_param_wave_correct(cv::detail::WaveCorrectKind wave_correct_kind = (cv::detail::WaveCorrectKind)-1)
            {
                if ((int)wave_correct_kind < 0)
                    return;

                std::vector<cv::Mat> rmats;
                for (size_t i = 0; i < m_camera_params.size(); ++i)
                    rmats.push_back(m_camera_params[i].R.clone());

                cv::detail::waveCorrect(rmats, wave_correct_kind);
                for (size_t i = 0; i < m_camera_params.size(); ++i)
                    m_camera_params[i].R = rmats[i];
            }

            void panorama_param_correct()
            {
                float seam_work_aspect = 1;
                float warp_scale = 0.0;
                std::vector<double> camera_focals;

                if (m_rotation_warper.get() == NULL)
                    set_warper("spherical");

                for (int i = 0; i < m_camera_params.size(); i++)
                    camera_focals.push_back(m_camera_params[i].focal);

                /*焦距排序*/
                std::sort(camera_focals.begin(), camera_focals.end());

                /*求中位焦距*/
                if (camera_focals.size() % 2 == 1)
                    warp_scale = static_cast<float>(camera_focals[camera_focals.size() / 2]);
                else
                    warp_scale = static_cast<float>(camera_focals[camera_focals.size() / 2 - 1] + camera_focals[camera_focals.size() / 2]) * 0.5f;

                m_rotation_warper->setScale(warp_scale);

                m_warper_xmaps.resize(m_img_good_sizes.size());
                m_warper_ymaps.resize(m_img_good_sizes.size());
                m_warper_rects.resize(m_img_good_sizes.size());

                for (int i = 0; i < m_img_good_sizes.size(); i++)
                {
                    cv::Mat_<float> K;
                    m_camera_params[i].K().convertTo(K, CV_32F);

                    K(0, 0) *= seam_work_aspect;
                    K(0, 2) *= seam_work_aspect;
                    K(1, 1) *= seam_work_aspect;
                    K(1, 2) *= seam_work_aspect;

                    m_warper_rects[i] = m_rotation_warper->buildMaps(m_img_good_sizes[i], K, m_camera_params[i].R, m_warper_xmaps[i], m_warper_ymaps[i]);

                    /*for @remap(+1,+1)*/
                    m_warper_rects[i].height += 1;
                    m_warper_rects[i].width += 1;
                }

                /*查找最小的X和Y。*/
                int min_x = m_warper_rects[0].x;
                int min_y = m_warper_rects[0].y;
                for (int i = 1; i < m_warper_rects.size(); i++)
                {
                    if (m_warper_rects[i].x < min_x)
                        min_x = m_warper_rects[i].x;

                    if (m_warper_rects[i].y < min_y)
                        min_y = m_warper_rects[i].y;
                }

                m_screen_rects.resize(m_warper_rects.size());

                /*相机坐标转屏幕坐标。*/
                for (int i = 0; i < m_warper_rects.size(); i++)
                {
                    m_screen_rects[i] = m_warper_rects[i];

                    if (min_x < 0)
                        m_screen_rects[i].x += abs(min_x);
                    else
                        m_screen_rects[i].x -= min_x;

                    if (min_y < 0)
                        m_screen_rects[i].y += abs(min_y);
                    else
                        m_screen_rects[i].y -= min_y;
                }

                m_blend_width = 0;
                m_blend_height = 0;

                /*计算拼接后图像的最大宽和高。*/
                for (int i = 0; i < m_screen_rects.size(); i++)
                {
                    if (m_blend_width < m_screen_rects[i].x + m_screen_rects[i].width)
                        m_blend_width = m_screen_rects[i].x + m_screen_rects[i].width;

                    if (m_blend_height < m_screen_rects[i].y + m_screen_rects[i].height)
                        m_blend_height = m_screen_rects[i].y + m_screen_rects[i].height;
                }

                m_blend_idxs = m_img_good_idxs;
                m_blend_rects = m_screen_rects;

                /*拼接前排序，横向。*/
                for (int i = 0; i < m_blend_idxs.size() - 1; i++)
                {
                    for (int j = i + 1; j < m_blend_idxs.size(); j++)
                    {
                        if (m_blend_rects[i].x > m_blend_rects[j].x)
                        {
                            std::swap(m_blend_rects[i], m_blend_rects[j]);
                            std::swap(m_blend_idxs[i], m_blend_idxs[j]);
                        }
                    }
                }
            }

            virtual bool remap(std::vector<cv::Mat> &outs, const std::vector<cv::Mat> &imgs)
            {
                assert(imgs.size() > 0);
                assert(imgs.size() >= m_img_good_idxs.size());
                assert(m_img_good_sizes.size() == m_img_good_idxs.size());
                assert(m_warper_rects.size() == m_img_good_idxs.size());
                assert(m_warper_xmaps.size() == m_img_good_idxs.size());
                assert(m_warper_ymaps.size() == m_img_good_idxs.size());

                /*输出的数量和顺序相同，但未能拼接的输出图像为空。*/
                assert(outs.size() == imgs.size());

                for (int i = 0; i < m_img_good_idxs.size(); i++)
                {
                    int idx = m_img_good_idxs[i];
                    int img_w = m_img_good_sizes[i].width;
                    int img_h = m_img_good_sizes[i].height;
                    int warper_w = m_warper_rects[i].width;
                    int warper_h = m_warper_rects[i].height;
                    auto &xmap_it = m_warper_xmaps[i];
                    auto &ymap_it = m_warper_ymaps[i];
                    auto &imgs_it = imgs[idx];
                    auto &outs_it = outs[idx];

                    assert(imgs_it.cols == img_w && imgs_it.rows == img_h);

                    /*创建变换后的图像存储空间。*/
                    outs_it.create(warper_h, warper_w, imgs_it.type());
                    if (outs_it.empty())
                        return false;

                    cv::remap(imgs_it, outs_it, xmap_it, ymap_it, cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                }

                return true;
            }

            virtual bool compose(cv::Mat &out, const std::vector<cv::Mat> &imgs, bool optimize_seam = true)
            {
                uint8_t scalar[4] = {0};

                assert(imgs.size() >= 0);

                /*创建全景图像存储空间。*/
                out.create(m_blend_height, m_blend_width, imgs[0].type());
                if (out.empty())
                    return false;

                for (int i = 0; i < m_blend_idxs.size(); i++)
                {
                    int idx = m_blend_idxs[i];
                    cv::Rect r = m_blend_rects[i];
                    auto &imgs_it = imgs[idx];

                    assert(imgs_it.type() == out.type());

                    /*计算重叠宽度。*/
                    int overlap_w = (i <= 0 ? 0 : (m_blend_rects[i - 1].width + m_blend_rects[i - 1].x - m_blend_rects[i].x));

                    abcdk::generic::imageproc::compose<uint8_t>(out.channels(), true,
                                                                out.data, out.cols, out.step, out.rows,
                                                                imgs_it.data, imgs_it.cols, imgs_it.step, imgs_it.rows,
                                                                scalar, r.x, r.y, overlap_w, (optimize_seam ? 1 : 0));
                }

                return true;
            }

        public:
            void set_feature_finder(cv::Ptr<cv::Feature2D> feature_finder)
            {
                m_feature_finder = feature_finder;
            }

            void set_feature_matcher(cv::Ptr<cv::detail::FeaturesMatcher> feature_matcher)
            {
                m_feature_matcher = feature_matcher;
            }

            void set_estimator(cv::Ptr<cv::detail::Estimator> estimator)
            {
                m_estimator = estimator;
            }

            void set_bundle_adjuster(cv::Ptr<cv::detail::BundleAdjusterBase> adjuster)
            {
                m_bundle_adjuster = adjuster;
            }

            void set_warper(cv::Ptr<cv::detail::RotationWarper> warper)
            {
                m_rotation_warper = warper;
            }

            void set_feature_finder(const char *name)
            {
                assert(name != NULL);

                if (strcasecmp(name, "ORB") == 0)
                    set_feature_finder(cv::ORB::create());
#ifdef OPENCV_ENABLE_NONFREE
                else if (strcasecmp(name, "SIFT") == 0)
                    set_feature_finder(cv::xfeatures2d::SIFT::create());
                else if (strcasecmp(name, "SURF") == 0)
                    set_feature_finder(cv::xfeatures2d::SURF::create());
#endif // OPENCV_ENABLE_NONFREE
                else
                {
                    abcdk_trace_printf(LOG_WARNING, "特征发现算法('%s')未找到，启用默认的算法('ORB')。", name);
                    set_feature_finder("ORB");
                }
            }

            void set_feature_matcher(const char *name, float match_conf = 0.3)
            {
                assert(name != NULL);

                if (strcasecmp(name, "Range") == 0)
                    set_feature_matcher(cv::makePtr<cv::detail::BestOf2NearestRangeMatcher>(5, false, match_conf));
                else if (strcasecmp(name, "Affine") == 0)
                    set_feature_matcher(cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>(false, false, match_conf));
                else if (strcasecmp(name, "Best") == 0)
                    set_feature_matcher(cv::makePtr<cv::detail::BestOf2NearestMatcher>(false, match_conf));
                else
                {
                    abcdk_trace_printf(LOG_WARNING, "特征匹配算法('%s')未找到，启用默认的算法('Best')。", name);
                    set_feature_matcher("Best");
                }
            }

            void set_estimator(const char *name)
            {
                assert(name != NULL);

                if (strcasecmp(name, "Affine") == 0)
                    set_estimator(cv::makePtr<cv::detail::AffineBasedEstimator>());
                else if (strcasecmp(name, "Homography") == 0)
                    set_estimator(cv::makePtr<cv::detail::HomographyBasedEstimator>());
                else
                {
                    abcdk_trace_printf(LOG_WARNING, "相机参数估计算法('%s')未找到，启用默认的算法('Homography')。", name);
                    set_estimator("Homography");
                }
            }

            void set_bundle_adjuster(const char *name)
            {
                assert(name != NULL);

                if (strcasecmp(name, "reproj") == 0)
                    set_bundle_adjuster(cv::makePtr<cv::detail::BundleAdjusterReproj>());
                else if (strcasecmp(name, "ray") == 0)
                    set_bundle_adjuster(cv::makePtr<cv::detail::BundleAdjusterRay>());
                else if (strcasecmp(name, "affine") == 0)
                    set_bundle_adjuster(cv::makePtr<cv::detail::BundleAdjusterAffinePartial>());
                else if (strcasecmp(name, "no") == 0)
                    set_bundle_adjuster(cv::makePtr<cv::detail::NoBundleAdjuster>());
                else
                {
                    abcdk_trace_printf(LOG_WARNING, "相机参数调节算法('%s')未找到，启用默认的算法('ray')。", name);
                    set_bundle_adjuster("ray");
                }
            }

            void set_warper(const char *name, float scale = 1.0)
            {
                assert(name != NULL);

                if (strcasecmp(name, "plane") == 0)
                    set_warper(cv::makePtr<cv::detail::PlaneWarper>(scale));
                else if (strcasecmp(name, "affine") == 0)
                    set_warper(cv::makePtr<cv::detail::AffineWarper>(scale));
                else if (strcasecmp(name, "cylindrical") == 0)
                    set_warper(cv::makePtr<cv::detail::CylindricalWarper>(scale));
                else if (strcasecmp(name, "spherical") == 0)
                    set_warper(cv::makePtr<cv::detail::SphericalWarper>(scale));
                else
                {
                    abcdk_trace_printf(LOG_WARNING, "图像变换算法('%s')未找到，启用默认的算法('spherical')。", name);
                    set_warper("spherical");
                }

                /*通知必须重新构建相机参数。*/
                m_panorama_param_ok = false;
            }

            void DrawKeypointsMatches(std::vector<cv::Mat> &outs, const std::vector<cv::Mat> &imgs)
            {
                draw_keypoints_matches(imgs, outs);
            }

            int EstimateTransform(const std::vector<cv::Mat> &imgs, const std::vector<cv::Mat> &masks,
                                  float good_threshold = 0.8, float adjuster_threshold = 0.8,
                                  cv::detail::WaveCorrectKind wave_correct_kind = (cv::detail::WaveCorrectKind)-1)
            {
                assert(imgs.size() >= 2);
                assert(masks.size() == 0 || imgs.size() == masks.size());

                assert(!m_camera_param_ok);

                find_feature(imgs, masks);

                if (!match_feature())
                    return -1;

                if (!leave_biggest_component(imgs, good_threshold))
                    return -2;

                if (!estimate_camera())
                    return -3;

                if (!camera_adjuster(adjuster_threshold))
                    return -4;

                camera_param_wave_correct(wave_correct_kind);

                m_camera_param_ok = true; // OK.

                return 0;
            }

            void BuildPanoramaParam()
            {
                assert(!m_panorama_param_ok);

                panorama_param_correct();

                m_panorama_param_ok = true; // OK.
            }

            int ComposePanorama(cv::Mat &out, const std::vector<cv::Mat> &imgs, bool optimize_seam = true)
            {
                std::vector<cv::Mat> remap_imgs;

                assert(imgs.size() >= 0);
                assert(imgs.size() >= m_img_good_idxs.size());

                assert(m_panorama_param_ok);

                remap_imgs.resize(imgs.size());

                if (!remap(remap_imgs, imgs))
                    return -1;

                if (!compose(out, remap_imgs, optimize_seam))
                    return -2;

                return 0;
            }
        };
    } //    namespace opencv
} // namespace abcdk

#endif // OPENCV_STITCHING_STITCHER_HPP

#endif // ABCDK_OPENCV_STITCHER_HXX