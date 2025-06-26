/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_RETINAFACE_FPN_HXX
#define ABCDK_TORCH_HOST_DNN_RETINAFACE_FPN_HXX

#include "dnn_model.hxx"

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class retinaface_fpn : public model
            {
            private:
                std::map<int, std::vector<float>> m_anchors;
                std::vector<float> m_variances;
                std::string m_cls_name_prefix;
                std::string m_box_name_prefix;
                std::string m_kpt_name_prefix;

            public:
                retinaface_fpn(const char *name = "")
                    : model(name)
                {
                }
                virtual ~retinaface_fpn()
                {
                }

            public:
                virtual void prepare(abcdk_option_t *opt)
                {
                    m_anchors[8].push_back(16);
                    m_anchors[8].push_back(32);
                    m_anchors[16].push_back(64);
                    m_anchors[16].push_back(128);
                    m_anchors[32].push_back(256);
                    m_anchors[32].push_back(512);

                    m_variances.push_back(0.1);
                    m_variances.push_back(0.2);

                    m_cls_name_prefix = abcdk_option_get(opt, "--output-cls-name-prefix", 0, "face_rpn_cls_prob_reshape_stride");
                    m_box_name_prefix = abcdk_option_get(opt, "--output-box-name-prefix", 0, "face_rpn_bbox_pred_stride");
                    m_kpt_name_prefix = abcdk_option_get(opt, "--output-kpt-name-prefix", 0, "face_rpn_landmark_pred_stride");
                }

                virtual void collect_object(std::vector<std::map<int, std::vector<object>>> &dst, std::vector<abcdk_torch_dnn_tensor_t> &tensor, float threshold)
                {
                    abcdk_torch_dnn_tensor_t *input_tensor_p = NULL;
                    abcdk_torch_dnn_tensor_t *output_cls_tensor_p = NULL;
                    abcdk_torch_dnn_tensor_t *output_box_tensor_p = NULL;
                    abcdk_torch_dnn_tensor_t *output_kpt_tensor_p = NULL;

                    for (auto &one_tensor : tensor)
                    {
                        if (one_tensor.mode == 1)
                        {
                            ABCDK_ASSERT(input_tensor_p == NULL, TT("RETINAFACE-FPN模型仅支持一个输入层。"));
                            input_tensor_p = &one_tensor;
                        }
                    }

                    ABCDK_ASSERT(input_tensor_p != NULL, TT("RETINAFACE-FPN模型至少存在一个输入层。"));

                    for (auto &one_anchor : m_anchors)
                    {
                        // 清空，为了下一个尺寸。
                        output_cls_tensor_p = output_box_tensor_p = output_kpt_tensor_p = NULL;

                        for (auto &one_tensor : tensor)
                        {
                            if (one_tensor.mode == 2)
                            {
                                std::string cls_name = m_cls_name_prefix + std::to_string(one_anchor.first);
                                std::string box_name = m_box_name_prefix + std::to_string(one_anchor.first);
                                std::string kpt_name = m_kpt_name_prefix + std::to_string(one_anchor.first);

                                if (abcdk_strcmp(one_tensor.name_p, cls_name.c_str(), 0) == 0)
                                {
                                    ABCDK_ASSERT(output_cls_tensor_p == NULL, TT("RETINAFACE-FPN模型同一个尺寸的CLS输出层仅支持一个。"));
                                    output_cls_tensor_p = &one_tensor;
                                }
                                else if (abcdk_strcmp(one_tensor.name_p, box_name.c_str(), 0) == 0)
                                {
                                    ABCDK_ASSERT(output_box_tensor_p == NULL, TT("RETINAFACE-FPN模型同一个尺寸的BOX输出层仅支持一个。"));
                                    output_box_tensor_p = &one_tensor;
                                }
                                else if (abcdk_strcmp(one_tensor.name_p, kpt_name.c_str(), 0) == 0)
                                {
                                    ABCDK_ASSERT(output_kpt_tensor_p == NULL, TT("RETINAFACE-FPN模型同一个尺寸的KPT输出层仅支持一个。"));
                                    output_kpt_tensor_p = &one_tensor;
                                }
                            }
                        }

                        ABCDK_ASSERT(output_cls_tensor_p != NULL, TT("RETINAFACE-FPN模型至少存在一个CLS输出层。"));
                        ABCDK_ASSERT(output_box_tensor_p != NULL, TT("RETINAFACE-FPN模型至少存在一个BOX输出层。"));
                        ABCDK_ASSERT(output_kpt_tensor_p != NULL, TT("RETINAFACE-FPN模型至少存在一个KPT输出层。"));

                        int input_b, input_c, input_h, input_w;
                        int output_cls_b, output_cls_c, output_cls_h, output_cls_w;
                        int output_box_b, output_box_c, output_box_h, output_box_w;
                        int output_kpt_b, output_kpt_c, output_kpt_h, output_kpt_w;

                        input_b = input_tensor_p->dims.d[0];
                        input_c = input_tensor_p->dims.d[1];
                        input_h = input_tensor_p->dims.d[2];
                        input_w = input_tensor_p->dims.d[3];

                        output_cls_b = output_cls_tensor_p->dims.d[0];
                        output_cls_c = output_cls_tensor_p->dims.d[1];
                        output_cls_h = output_cls_tensor_p->dims.d[2];
                        output_cls_w = output_cls_tensor_p->dims.d[3];

                        output_box_b = output_box_tensor_p->dims.d[0];
                        output_box_c = output_box_tensor_p->dims.d[1];
                        output_box_h = output_box_tensor_p->dims.d[2];
                        output_box_w = output_box_tensor_p->dims.d[3];

                        output_kpt_b = output_kpt_tensor_p->dims.d[0];
                        output_kpt_c = output_kpt_tensor_p->dims.d[1];
                        output_kpt_h = output_kpt_tensor_p->dims.d[2];
                        output_kpt_w = output_kpt_tensor_p->dims.d[3];
                        

                        assert(input_b == output_cls_b && input_b == output_box_b && input_b == output_kpt_b);
                        assert(output_cls_h == output_box_h && output_box_h == output_kpt_h);
                        assert(output_cls_w == output_box_w && output_box_w == output_kpt_w);

                        assert(output_cls_c / 2 == one_anchor.second.size()); // 输出层里包括背景和前景，背景在推理时使用，后处理时不需要。

#if 1
                        abcdk::torch::ndarray output_cls_data((void *)output_cls_tensor_p->data_p, false, output_cls_b, output_cls_c, output_cls_h, output_cls_w, output_cls_w * sizeof(float));
                        abcdk::torch::ndarray output_box_data((void *)output_box_tensor_p->data_p, false, output_box_b, output_box_c, output_box_h, output_box_w, output_box_w * sizeof(float));
                        abcdk::torch::ndarray output_kpt_data((void *)output_kpt_tensor_p->data_p, false, output_kpt_b, output_kpt_c, output_kpt_h, output_kpt_w, output_kpt_w * sizeof(float));
#else 
                        abcdk::torch::ndarray output_cls_data((void *)output_cls_tensor_p->data_p, true, output_cls_b, output_cls_c, output_cls_h, output_cls_w, output_cls_w * output_cls_c * sizeof(float));
                        abcdk::torch::ndarray output_box_data((void *)output_box_tensor_p->data_p, true, output_box_b, output_box_c, output_box_h, output_box_w, output_box_w * output_box_c * sizeof(float));
                        abcdk::torch::ndarray output_kpt_data((void *)output_kpt_tensor_p->data_p, true, output_kpt_b, output_kpt_c, output_kpt_h, output_kpt_w, output_kpt_w * output_kpt_c * sizeof(float));
#endif 

                        int output_classes = one_anchor.second.size();

                        dst.clear();
                        dst.resize(output_cls_b);

                        for (int b = 0; b < output_cls_b; b++)
                        {
                            for (int y = 0; y < output_cls_h; y++)
                            {
                                for (int x = 0; x < output_cls_w; x++)
                                {
                                    /*b,c,b,c,[b,c,b,c,...]*/

                                    for (int c = 0; c < output_classes; c++)
                                    {
                                        float score = output_cls_data.obj<float>(b, x, y,  c * output_classes + 1);

                                        /*低于阈值的不要。*/
                                        if (score < threshold)
                                            continue;

                                        
                                        abcdk::torch_host::dnn::object one_dst;

                                        one_dst.m_label = 1;
                                        one_dst.m_score = (int)(score * 100);

                                        float anchor_cx = ((float)x + 0.5) * one_anchor.first; // 计算当前网格单元格的中心X坐标。
                                        float anchor_cy = ((float)y + 0.5) * one_anchor.first; // 计算当前网格单元格的中心Y坐标。
                                        float anchor_w = one_anchor.second[c]; 
                                        float anchor_h = one_anchor.second[c];

                                        float _x = output_box_data.obj<float>(b, x, y, c * 4 + 0); // 矩开中心点X坐标。
                                        float _y = output_box_data.obj<float>(b, x, y, c * 4 + 1); // 矩开中心点Y坐标。
                                        float _w = output_box_data.obj<float>(b, x, y, c * 4 + 2);
                                        float _h = output_box_data.obj<float>(b, x, y, c * 4 + 3);

                                        float predicted_cx = anchor_cx + _x * anchor_w * m_variances[0];
                                        float predicted_cy = anchor_cy + _y * anchor_h * m_variances[0];
                                        float predicted_w = anchor_w * std::exp(_w * m_variances[1]);
                                        float predicted_h = anchor_h * std::exp(_h * m_variances[1]);

                                        one_dst.m_rect_x1 = (predicted_cx - predicted_w / 2);
                                        one_dst.m_rect_y1 = (predicted_cy - predicted_h / 2);
                                        one_dst.m_rect_x2 = (predicted_cx + predicted_w / 2);
                                        one_dst.m_rect_y2 = (predicted_cy + predicted_h / 2);

                                        /*修正坐标，不要超出图像范围。*/
                                        one_dst.m_rect_x1 = abcdk::torch::util::clamp<int>(one_dst.m_rect_x1, 0, input_w - 1);
                                        one_dst.m_rect_y1 = abcdk::torch::util::clamp<int>(one_dst.m_rect_y1, 0, input_h - 1);
                                        one_dst.m_rect_x2 = abcdk::torch::util::clamp<int>(one_dst.m_rect_x2, 0, input_w - 1);
                                        one_dst.m_rect_y2 = abcdk::torch::util::clamp<int>(one_dst.m_rect_y2, 0, input_h - 1);

                                        dst[b][one_dst.m_label].push_back(one_dst);

                                    }
                                }
                            }
                        }
                    }
                }
            };
        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ABCDK_TORCH_HOST_DNN_RETINAFACE_FPN_HXX