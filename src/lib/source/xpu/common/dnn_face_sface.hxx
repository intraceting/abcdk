/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_DNN_FACE_SFACE_HXX
#define ABCDK_XPU_COMMON_DNN_FACE_SFACE_HXX

#include "dnn_model.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace dnn
        {
            class face_sface : public model
            {
            public:
                face_sface(const char *name = "")
                    : model(name)
                {
                }
                virtual ~face_sface()
                {
                }

            public:
                virtual void prepare(abcdk_option_t *opt)
                {
                }

            protected:
                virtual void collect_object(std::vector<std::map<int, std::vector<object>>> &dst, std::vector<abcdk_xpu_dnn_tensor_t> &tensor, float threshold)
                {
                    abcdk_xpu_dnn_tensor_t *input_tensor_p = NULL, *output_tensor_p = NULL;

                    for (auto &t : tensor)
                    {
                        if (t.mode == 1)
                        {
                            ABCDK_TRACE_ASSERT(input_tensor_p == NULL, ABCDK_GETTEXT("FACE-SFACE模型仅支持一个输入层."));
                            input_tensor_p = &t;
                        }
                        else if (t.mode == 2)
                        {
                            ABCDK_TRACE_ASSERT(output_tensor_p == NULL, ABCDK_GETTEXT("FACE-SFACE模型仅支持一个输出层."));
                            assert(output_tensor_p == NULL);
                            output_tensor_p = &t;
                        }
                        else
                            continue;
                    }

                    ABCDK_TRACE_ASSERT(input_tensor_p != NULL, ABCDK_GETTEXT("FACE-SFACE模型至少存在一个输入层."));
                    ABCDK_TRACE_ASSERT(output_tensor_p != NULL, ABCDK_GETTEXT("FACE-SFACE模型至少存在一个输出层."));

                    int input_b, input_c, input_h, input_w;
                    int output_b, output_c, output_h, output_w;

                    input_b = input_tensor_p->dims.d[0];
                    input_c = input_tensor_p->dims.d[1];
                    input_h = input_tensor_p->dims.d[2];
                    input_w = input_tensor_p->dims.d[3];

                    output_b = output_tensor_p->dims.d[0];
                    output_c = output_tensor_p->dims.d[1];
                    output_h = 1; // output_tensor_p->dims.d[2];
                    output_w = 1; // output_tensor_p->dims.d[3];

                    ndarray output_data((void *)output_tensor_p->data_p, false, output_b, output_c, output_h, output_w, output_w * sizeof(float));

                    dst.clear();
                    dst.resize(output_b);

                    for (int b = 0; b < output_b; b++)
                    {
                        int label = 0;

                        object one_dst;

                        one_dst.m_label = label;
                        one_dst.m_score = 99;

                        one_dst.m_rect_x1 = (int)(0);
                        one_dst.m_rect_y1 = (int)(0);
                        one_dst.m_rect_x2 = (int)(input_h - 1);
                        one_dst.m_rect_y2 = (int)(input_w - 1);

                        one_dst.m_feature.resize(output_c);

                        memcpy(one_dst.m_feature.data(),output_data.ptr<float>(b,0,0,0),output_c);

                        dst[b][label].push_back(one_dst);
                    }
                }
            };

        } // namespace dnn
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_DNN_FACE_SFACE_HXX