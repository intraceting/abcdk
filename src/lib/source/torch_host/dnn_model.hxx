/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_MODEL_HXX
#define ABCDK_TORCH_HOST_DNN_MODEL_HXX

#include "abcdk/util/math.h"
#include "abcdk/torch/opencv.h"
#include "../torch/util.hxx"
#include "../torch/ndarray.hxx"
#include "dnn_object.hxx"

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class model
            {
            public:
                /**
                 * @param dst <object>
                 * @param src <object>
                 */
                static inline void nms(std::vector<object> &dst, std::vector<object> &src, float threshold)
                {
                    /*按分数排序(降序)*/
                    std::sort(src.begin(), src.end(), [](object &b1, object &b2)
                              { return b1.m_score > b2.m_score; });

                    for (size_t i = 0; i < src.size(); i++)
                    {
                        bool keep = true;
                        double overlap = 0.0;
                        for (size_t j = 0; j < dst.size(); j++)
                        {
                            overlap = dst[j].overlap_ratio(src[i]);

                            keep = (overlap < threshold);
                            if (!keep)
                                break;
                        }

                        if (keep)
                            dst.push_back(src[i]);
                    }
                }

                /**
                 * @param dst <object>
                 * @param src <label<object>>
                 */
                static inline void nms(std::vector<object> &dst, std::map<int, std::vector<object>> &src, float threshold)
                {
                    dst.clear();

                    /*按KEY分别做NMS。*/
                    for (auto &t : src)
                    {
                        std::vector<object> tmp_dst;
                        nms(tmp_dst, t.second, threshold);

                        for (auto &t2 : tmp_dst)
                        {
                            dst.push_back(t2);
                        }
                    }
                }

#ifdef OPENCV_CORE_HPP
                static inline void nms_rotated(std::vector<object> &dst, std::vector<object> &src, float threshold)
                {
                    /*按分数排序(降序)*/
                    std::sort(src.begin(), src.end(), [](object &b1, object &b2)
                              { return b1.m_score > b2.m_score; });

                    std::vector<float> vArea(src.size());

                    /*计算每个旋转矩形的面积*/
                    for (int i = 0; i < int(src.size()); ++i)
                    {
                        vArea[i] = src[i].rrect().size.area();
                    }

                    std::vector<bool> isSuppressed(src.size(), false);

                    for (int i = 0; i < int(src.size()); ++i)
                    {
                        if (isSuppressed[i])
                            continue;

                        for (int j = i + 1; j < int(src.size()); ++j)
                        {
                            if (isSuppressed[j])
                                continue;

                            std::vector<cv::Point2f> intersectingRegion;

                            // 返回两个旋转矩形之间相交区域的顶点
                            cv::rotatedRectangleIntersection(src[i].rrect(), src[j].rrect(), intersectingRegion);
                            if (intersectingRegion.empty())
                                continue;

                            // 计算两个旋转矩形相交区域的面积
                            float inter = cv::contourArea(intersectingRegion);
                            if (src[i].m_label == src[j].m_label)
                            {
                                float ovr = inter / (vArea[i] + vArea[j] - inter); // 计算iou
                                if (ovr >= threshold)
                                    isSuppressed[j] = true;
                            }
                        }
                    }

                    for (int i = 0; i < src.size(); i++)
                    {
                        if (!isSuppressed[i])
                            dst.push_back(src[i]);
                    }
                }
#endif // OPENCV_CORE_HPP

                /**
                 * @param dst <object>
                 * @param src <label<object>>
                 */
                static inline void nms_rotated(std::vector<object> &dst, std::map<int, std::vector<object>> &src, float threshold)
                {
                    dst.clear();

                    /*按KEY分别做NMS。*/
                    for (auto &t : src)
                    {
                        std::vector<object> tmp_dst;
#ifdef OPENCV_CORE_HPP
                        nms_rotated(tmp_dst, t.second, threshold);
#endif // OPENCV_CORE_HPP

                        for (auto &t2 : tmp_dst)
                        {
                            dst.push_back(t2);
                        }
                    }
                }

            private:
                /**/
                std::string m_name;

                //<batch<object>>
                std::vector<std::vector<object>> m_object;

            public:
                model(const char *name = "")
                {
                    m_name = (name ? name : "");
                }
                
                virtual ~model()
                {

                }
                
            public:
                const char *name()
                {
                    return m_name.c_str();
                }

            public:

                virtual void prepare(abcdk_option_t *opt)
                {

                }

                virtual void process(std::vector<abcdk_torch_dnn_tensor> &tensor, float score_threshold, float nms_threshold)
                {
                    std::vector<std::map<int, std::vector<object>>> src;

                    collect_object(src, tensor, score_threshold);

                    nms_object(m_object, src, nms_threshold);
                }

                virtual void fetch(std::vector<abcdk_torch_dnn_object_t> &dst, int index)
                {
                    ABCDK_ASSERT(index < m_object.size(), TT("超出图像批量范围。"));

                    dst.clear();
                    dst.resize(m_object[index].size());

                    for (int i = 0; i < m_object[index].size(); i++)
                    {
                        dst[i].label = m_object[index][i].m_label;

                        dst[i].score = m_object[index][i].m_score;

                        dst[i].rect.nb = 2;
                        dst[i].rect.pt[0].x = m_object[index][i].m_rect_x1;
                        dst[i].rect.pt[0].y = m_object[index][i].m_rect_y1;
                        dst[i].rect.pt[1].x = m_object[index][i].m_rect_x2;
                        dst[i].rect.pt[1].y = m_object[index][i].m_rect_y2;

                        dst[i].angle = m_object[index][i].m_angle;
#ifdef OPENCV_CORE_HPP
                        cv::Point2f vec_pts[4];
                        m_object[index][i].rrect2pts(vec_pts);

                        dst[i].rrect.nb = 4;
                        for (int j = 0; j < 4; j++)
                        {
                            dst[i].rrect.pt[j].x = vec_pts[j].x;
                            dst[i].rrect.pt[j].y = vec_pts[j].y;
                        }
#endif // OPENCV_CORE_HPP

                        dst[i].nkeypoint = m_object[index][i].m_keypoint.size() / 3;
                        dst[i].kp = m_object[index][i].m_keypoint.data();

                        dst[i].nfeature = m_object[index][i].m_feature.size();
                        dst[i].ft = m_object[index][i].m_feature.data();

                        dst[i].seg_step = m_object[index][i].m_seg_step;
                        dst[i].seg = m_object[index][i].m_segment.data();

                    }
                }

            protected:

                /**
                 * @param dst <batch<label<object>>>
                 */
                virtual void collect_object(std::vector<std::map<int, std::vector<object>>> &dst, std::vector<abcdk_torch_dnn_tensor> &tensor, float threshold) = 0;

                /**
                 * @param dst <batch<object>>
                 * @param src <batch<label<object>>>
                 */
                virtual void nms_object(std::vector<std::vector<object>> &dst, std::vector<std::map<int, std::vector<object>>> &src, float threshold)
                {
                    dst.resize(src.size());
                    for (int i = 0; i < src.size(); i++)
                    {
                        nms(dst[i], src[i], threshold);
                    }
                }

            };
        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ABCDK_TORCH_HOST_DNNPOST_MODEL_HXX