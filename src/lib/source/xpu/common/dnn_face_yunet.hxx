/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_DNN_FACE_YUNET_HXX
#define ABCDK_XPU_COMMON_DNN_FACE_YUNET_HXX

#include "dnn_model.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace dnn
        {
            class face_yunet : public model
            {
            private:
                int m_output_kpt_2d;
                int m_output_kpt_num;
            public:
                face_yunet(const char *name = "")
                    : model(name)
                {

                }
                virtual ~face_yunet()
                {
                }

            public:
                virtual void prepare(abcdk_option_t *opt)
                {
                    m_output_kpt_2d = abcdk_option_get_int(opt, "--output-keypoint-dims-2d", 0, 1);
                    m_output_kpt_num = abcdk_option_get_int(opt, "--output-keypoint-number", 0, 5);
                }

            protected:
                abcdk_xpu_dnn_tensor_t *find_input_tensor(std::vector<abcdk_xpu_dnn_tensor_t> &tensor)
                {
                    for (auto &one : tensor)
                    {
                        if (one.mode == 1)
                            return &one;
                    }

                    return NULL;
                }

                abcdk_xpu_dnn_tensor_t *find_output_tensor(std::vector<abcdk_xpu_dnn_tensor_t> &tensor, const char *prefix, int stride)
                {
                    std::string name = std::string(prefix) + std::to_string(stride);

                    for (auto &one : tensor)
                    {
                        if (one.mode != 2)
                            continue;

                        if (name.compare(one.name_p) == 0)
                            return &one;
                    }

                    return NULL;
                }

            protected:
                virtual void collect_object(std::vector<std::map<int, std::vector<object>>> &dst, std::vector<abcdk_xpu_dnn_tensor_t> &tensor, float threshold)
                {
                    abcdk_xpu_dnn_tensor_t *input_tensor_p = NULL;
                    abcdk_xpu_dnn_tensor_t *output_tensor_cls_p = NULL;
                    abcdk_xpu_dnn_tensor_t *output_tensor_obj_p = NULL;
                    abcdk_xpu_dnn_tensor_t *output_tensor_box_p = NULL;
                    abcdk_xpu_dnn_tensor_t *output_tensor_kps_p = NULL;

                    input_tensor_p = find_input_tensor(tensor);

                    ABCDK_TRACE_ASSERT(input_tensor_p != NULL, ABCDK_GETTEXT("FACE-YUNET模型至少存在一个输入层."));

                    int input_b, input_c, input_h, input_w;
                    int output_b, output_c, output_h, output_w;

                    input_b = input_tensor_p->dims.d[0];
                    input_c = input_tensor_p->dims.d[1];
                    input_h = input_tensor_p->dims.d[2];
                    input_w = input_tensor_p->dims.d[3];

                    int output_kpt_size = (m_output_kpt_2d ? 2 : 3);

                    std::vector<int> strides = {8, 16, 32};

                    dst.clear();

                    for (size_t i = 0; i < strides.size(); ++i)
                    {
                        int anchor_size = strides[i];

                        output_tensor_cls_p = find_output_tensor(tensor, "cls_", anchor_size);
                        output_tensor_obj_p = find_output_tensor(tensor, "obj_", anchor_size);
                        output_tensor_box_p = find_output_tensor(tensor, "bbox_", anchor_size);
                        output_tensor_kps_p = find_output_tensor(tensor, "kps_", anchor_size);

                        ABCDK_TRACE_ASSERT(output_tensor_cls_p != NULL, ABCDK_GETTEXT("FACE-YUNET模型至少存在一个CLS输出层."));
                        ABCDK_TRACE_ASSERT(output_tensor_obj_p != NULL, ABCDK_GETTEXT("FACE-YUNET模型至少存在一个OBJ输出层."));
                        ABCDK_TRACE_ASSERT(output_tensor_box_p != NULL, ABCDK_GETTEXT("FACE-YUNET模型至少存在一个BOX输出层."));
                        ABCDK_TRACE_ASSERT(output_tensor_kps_p != NULL, ABCDK_GETTEXT("FACE-YUNET模型至少存在一个KPS输出层."));

                        output_b = output_tensor_cls_p->dims.d[0];
                        // output_c = output_tensor_cls_p->dims.d[2];
                        output_h = int(input_h / anchor_size);
                        output_w = int(input_w / anchor_size);

                        ndarray cls_data((void *)output_tensor_cls_p->data_p, true, output_tensor_cls_p->dims.d[0], output_tensor_cls_p->dims.d[2], output_h, output_w, output_w * output_tensor_cls_p->dims.d[2] * sizeof(float));
                        ndarray obj_data((void *)output_tensor_obj_p->data_p, true, output_tensor_obj_p->dims.d[0], output_tensor_obj_p->dims.d[2], output_h, output_w, output_w * output_tensor_obj_p->dims.d[2] * sizeof(float));
                        ndarray box_data((void *)output_tensor_box_p->data_p, true, output_tensor_box_p->dims.d[0], output_tensor_box_p->dims.d[2], output_h, output_w, output_w * output_tensor_box_p->dims.d[2] * sizeof(float));
                        ndarray kps_data((void *)output_tensor_kps_p->data_p, true, output_tensor_kps_p->dims.d[0], output_tensor_kps_p->dims.d[2], output_h, output_w, output_w * output_tensor_kps_p->dims.d[2] * sizeof(float));

                        /*仅需初始一次.*/
                        if (dst.size() != output_b)
                            dst.resize(output_b);

                        for (int b = 0; b < output_b; b++)
                        {
                            for (int y = 0; y < output_h; y++)
                            {
                                for (int x = 0; x < output_w; x++)
                                {
                                    int label = 0;

                                    float cls_score = cls_data.obj<float>(b, x, y, 0);
                                    float obj_score = obj_data.obj<float>(b, x, y, 0);

                                    cls_score = util::clamp<float>(cls_score, 0.f, 1.f);
                                    obj_score = util::clamp<float>(obj_score, 0.f, 1.f);

                                    /*相乘,然后再开平方.*/
                                    float max_score = std::sqrt(cls_score * obj_score);

                                    /*低于阈值的不要.*/
                                    if (max_score < threshold)
                                        continue;

                                    float cx = (x + box_data.obj<float>(b, x, y, 0)) * anchor_size; // 矩型中心点X坐标.
                                    float cy = (y + box_data.obj<float>(b, x, y, 1)) * anchor_size; // 矩型中心点Y坐标.
                                    float _w = exp(box_data.obj<float>(b, x, y, 2)) * anchor_size;
                                    float _h = exp(box_data.obj<float>(b, x, y, 3)) * anchor_size;

                                    object one_dst;

                                    one_dst.m_label = label;
                                    one_dst.m_score = (int)(max_score * 100);

                                    one_dst.m_rect_x1 = (int)(cx - _w / 2);
                                    one_dst.m_rect_y1 = (int)(cy - _h / 2);
                                    one_dst.m_rect_x2 = (int)(cx + _w / 2);
                                    one_dst.m_rect_y2 = (int)(cy + _h / 2);

                                    /*修正坐标，不要超出图像范围.*/
                                    one_dst.m_rect_x1 = util::clamp<int>(one_dst.m_rect_x1, 0, input_w - 1);
                                    one_dst.m_rect_y1 = util::clamp<int>(one_dst.m_rect_y1, 0, input_h - 1);
                                    one_dst.m_rect_x2 = util::clamp<int>(one_dst.m_rect_x2, 0, input_w - 1);
                                    one_dst.m_rect_y2 = util::clamp<int>(one_dst.m_rect_y2, 0, input_h - 1);

                                    one_dst.m_keypoint.resize(m_output_kpt_num * 3);

                                    for (int n = 0, k = 0; n < m_output_kpt_num * output_kpt_size; n += output_kpt_size, k += 3)
                                    {
                                        one_dst.m_keypoint[k + 0] = (x + kps_data.obj<float>(b, x, y, n + 0)) * anchor_size;
                                        one_dst.m_keypoint[k + 1] = (y + kps_data.obj<float>(b, x, y, n + 1)) * anchor_size;
                                        one_dst.m_keypoint[k + 2] = 99;
                                    }

                                    dst[b][label].push_back(one_dst);
                                }
                            }
                        }
                    }
                }
            };

        } // namespace dnn
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_DNN_FACE_YUNET_HXX