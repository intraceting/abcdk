/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_DNN_OBJECT_HXX
#define ABCDK_TORCH_DNN_OBJECT_HXX

#include "abcdk/torch/dnn.h"
#include "abcdk/torch/opencv.h"

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

                /*关键点。x,y,v*/
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
                    m_seg_step = 0;
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

                double cx() const
                {
                    return x() + w() / 2;
                }
        
                double cy() const
                {
                    return y() + h() / 2;
                }

#ifdef OPENCV_CORE_HPP
                cv::RotatedRect rrect()
                {
                    cv::RotatedRect rb(cv::Point2f(cx(), cy()), cv::Size(w(), h()), m_rotate);
        
                    return rb;
                }
#endif //OPENCV_CORE_HPP

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