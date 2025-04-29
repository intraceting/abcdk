/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_MODEL_HXX
#define ABCDK_TORCH_MODEL_HXX

#include "invoke.hxx"
#include "bbox.hxx"

#include <vector>
#include <algorithm>
#include <map>

namespace abcdk
{
    namespace torch
    {
        class model
        {
        public:
            static inline void nms_iou(std::vector<bbox> &dst, std::vector<bbox> &src, float threshold)
            {
                /*按分数排序(降序)*/
                std::sort(src.begin(), src.end(), [](bbox &b1, bbox &b2)
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

            static inline void nms_iou(std::vector<bbox> &dst, std::map<int, std::vector<bbox>> &src, float threshold)
            {
                /*按KEY分别做NMS。*/
                for (auto &t : src)
                {
                    std::vector<bbox> tmp_dst;
                    nms_iou(tmp_dst, t.second, threshold);

                    for (auto &t2 : tmp_dst)
                    {
                        dst.push_back(t2);
                    }
                }
            }

        public:
            std::vector<bbox> m_boxs;

        public:
            model()
            {
            }
            virtual ~model()
            {
            }

        public:
            virtual void nms_filter(std::vector<bbox> &dst, const std::map<int,std::vector<bbox>> &src, float threshold)
            {
                nms_iou(dst,src,threshold);
            }

            virtual void get_bbox(std::vector<std::vector<bbox>> &dst, float score_threshold,float nms_threshold)
            {
                //<batch<label<bbox>>>
                std::vector<std::map<int,std::vector<bbox>>> src;

                get_bbox(src,score_threshold);

                //<batch<bbox>>
                dst.resize(src.size());
                for (int i = 0; i < src.size(); i++)
                {
                    nms_filter(dst[i], src[i], nms_threshold);
                }
            }

            virtual void get_bbox(std::vector<std::map<int,std::vector<bbox>>> &dst, float threshold)
            {

            }
        }
    } // namespace torch
} // namespace abcdk

#endif // ABCDK_TORCH_MODEL_HXX