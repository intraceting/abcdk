/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_DNN_OBJECT_HXX
#define ABCDK_TORCH_DNN_OBJECT_HXX

#include "abcdk/torch/dnn.h"

#include <vector>
#include <algorithm>
#include <map>

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class object
            {
            public:
                /**
                 * @param dst <object>
                 * @param src <object>
                 */
                static inline void nms_iou(std::vector<object> &dst, std::vector<object> &src, float threshold)
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
                static inline void nms_iou(std::vector<object> &dst, std::map<int, std::vector<object>> &src, float threshold)
                {
                    /*按KEY分别做NMS。*/
                    for (auto &t : src)
                    {
                        std::vector<object> tmp_dst;
                        nms_iou(tmp_dst, t.second, threshold);

                        for (auto &t2 : tmp_dst)
                        {
                            dst.push_back(t2);
                        }
                    }
                }

                /**
                 * @param dst <batch<object>>
                 * @param src <batch<label<object>>>
                 */
                static inline void nms_iou(std::vector<std::vector<object>> &dst, std::vector<std::map<int, std::vector<object>>> &src, float threshold)
                {
                    dst.resize(src.size());
                    for (int i = 0; i < src.size(); i++)
                    {
                        nms_iou(dst[i], src[i], threshold);
                    }
                }

            public:
                /*标签。*/
                int m_label;

                /*评分。0 ~ 99 */
                int m_score;

                /*矩形坐标。*/
                int m_rect_x1;
                int m_rect_y1;
                int m_rect_x2;
                int m_rect_y2;

                /*旋转角度。-90 ~ 90 */
                int m_rotate;

                /*关键点。x,y*/
                std::vector<int> m_keypoint;

                /*特征。*/
                std::vector<float> m_feature;

                /*分割。*/
                int m_seg_step;
                std::vector<uint8_t> m_segment;

            public:
                object()
                {
                    m_label = 0;
                    m_score = 0;
                    m_rect_x1 = -1;
                    m_rect_y1 = -1;
                    m_rect_x2 = -1;
                    m_rect_y2 = -1;
                    m_rotate = 0;
                }

                object(const object &src)
                {
                    *this = src;
                }

                virtual ~object()
                {
                }

            public:
                int x() const
                {
                    return m_rect_x1;
                }

                int y() const
                {
                    return m_rect_y1;
                }

                int w() const
                {
                    return m_rect_x2 - m_rect_x1;
                }

                int h() const
                {
                    return m_rect_y2 - m_rect_y1;
                }

            public:
                object &operator=(const object &src)
                {
                    /*如是自已，则忽略。*/
                    if (this == &src)
                        return *this;

                    m_label = src.m_label;
                    m_score = src.m_score;
                    m_rect_x1 = src.m_rect_x1;
                    m_rect_y1 = src.m_rect_y1;
                    m_rect_x2 = src.m_rect_x2;
                    m_rect_y2 = src.m_rect_y2;
                    m_rotate = src.m_rotate;
                    m_keypoint = src.m_keypoint;
                    m_feature = src.m_feature;
                    m_seg_step = src.m_seg_step;
                    m_segment = src.m_segment;

                    return *this;
                }

                /**计算两个矩型相交的面积。*/
                double overlap_area(const object &src)
                {
                    /*这四种情况都不相交。*/
                    if (x() > src.x() + src.w())
                        return 0.0;

                    if (y() > src.y() + src.h())
                        return 0.0;

                    if (x() + w() < src.x())
                        return 0.0;

                    if (y() + h() < src.y())
                        return 0.0;

                    /*有交集。*/
                    double cols = std::min(x() + w(), src.x() + src.w()) - std::max(x(), src.x());
                    double rows = std::min(y() + h(), src.y() + src.h()) - std::max(y(), src.y());
                    double overlap = cols * rows;

                    return overlap;
                }

                /**计算两个矩型相交的面积与两个矩型的面积之和比。*/
                double overlap_ratio(const object &src)
                {
                    double area_overlap = overlap_area(src);

                    double dst_area = w() * h();
                    double src_area = src.w() * src.h();
                    return area_overlap / (dst_area + src_area - area_overlap);
                }
            };
        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ABCDK_TORCH_DNN_OBJECT_HXX