/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_MODEL_HXX
#define ABCDK_TORCH_HOST_DNN_MODEL_HXX

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

            private:
                std::vector<std::vector<object>> m_object;

            public:
                model()
                {
                }
                virtual ~model()
                {
                }
            public:

                virtual void collect_object(std::vector<abcdk_torch_dnn_tensor> &tensor, float score_threshold, float nms_threshold)
                {
                    //<batch<label<object>>>
                    std::vector<std::map<int, std::vector<object>>> tmp;

                    get_object(tmp, score_threshold);

                    //<batch<object>>
                    m_object.resize(tmp.size());
                    for (int i = 0; i < tmp.size(); i++)
                    {
                        nms_iou(m_object[i], tmp[i], nms_threshold);
                    }
                }

                virtual void collect_object(std::vector<abcdk_torch_dnn_tensor> &tensor, float threshold)
                {

                }
            };
        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ABCDK_TORCH_HOST_DNNPOST_MODEL_HXX