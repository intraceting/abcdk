/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_YOLO_V11_POSE_HXX
#define ABCDK_TORCH_HOST_DNN_YOLO_V11_POSE_HXX

#include "dnn_model.hxx"

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class yolo_v11_pose : public model
            {
            private:
                int m_output_kpt_2d;
                int m_output_kpt_num;

            public:
                yolo_v11_pose(const char *name = "")
                    : model(name)
                {
                }
                virtual ~yolo_v11_pose()
                {
                }

            public:
                virtual void prepare(abcdk_option_t *opt)
                {
                    m_output_kpt_2d = abcdk_option_get_int(opt, "--output-keypoint-dims-2d", 0, 0);
                    m_output_kpt_num = abcdk_option_get_int(opt, "--output-keypoint-number", 0, 17);
                }

            protected:
                virtual void collect_object(std::vector<std::map<int, std::vector<object>>> &dst, std::vector<abcdk_torch_dnn_tensor> &tensor, float threshold)
                {
                    abcdk_torch_dnn_tensor *input_tensor_p = NULL, *output_tensor_p = NULL;

                    //<batch<label<object>>>
                    std::vector<std::map<int, std::vector<object>>> tmp_object;

                    for (auto &t : tensor)
                    {
                        if (t.mode == 1)
                        {
                            ABCDK_ASSERT(input_tensor_p == NULL, TT("YOLO-11模型仅支持一个输入层。"));
                            input_tensor_p = &t;
                        }
                        else if (t.mode == 2)
                        {
                            ABCDK_ASSERT(output_tensor_p == NULL, TT("YOLO-11模型仅支持一个输出层。"));
                            assert(output_tensor_p == NULL);
                            output_tensor_p = &t;
                        }
                        else
                            continue;
                    }

                    ABCDK_ASSERT(input_tensor_p != NULL, TT("YOLO-11模型至少存在一个输入层。"));
                    ABCDK_ASSERT(output_tensor_p != NULL, TT("YOLO-11模型至少存在一个输出层。"));

                    int input_b, input_c, input_h, input_w;
                    int output_b, output_c, output_h, output_w;
                    void *output_data;
                    int output_packed = 0;
                    int output_classes = 1;
                    int output_kpt_size = 3;

                    input_b = input_tensor_p->dims.d[0];
                    input_c = input_tensor_p->dims.d[1];
                    input_h = input_tensor_p->dims.d[2];
                    input_w = input_tensor_p->dims.d[3];

                    output_b = output_tensor_p->dims.d[0];
                    output_c = output_tensor_p->dims.d[1];
                    output_h = output_tensor_p->dims.d[2];
                    output_w = 1; // output_tensor_p->dims.d[3];

                    output_data = (void *)output_tensor_p->data_p;

                    assert(input_b == output_b);
                    assert(output_c >= 5);

                    output_kpt_size = (m_output_kpt_2d ? 2 : 3);
                    output_classes = output_c - 4 - (output_kpt_size * m_output_kpt_num);

                    dst.clear();
                    dst.resize(output_b);

                    for (int b = 0; b < output_b; b++)
                    {
                        for (int y = 0; y < output_h; y++)
                        {
                            int label = -1;
                            float max_score = 0.0;

                            /*
                             * 2D
                             * cx,cy,w,h,c0,[c1,...],[x1,y1,...]
                             * 
                             * 3D
                             * cx,cy,w,h,c0,[c1,...],[x1,y1,v1,...]
                             */

                            /*在所有分类中找出最大的。*/
                            for (int c = 0; c < output_classes; c++)
                            {
                                size_t off = abcdk::torch::util::off<float>(output_packed, output_w, output_w * sizeof(float), output_h, output_c, b, 0, y, 4 + c);
                                float score = abcdk::torch::util::obj<float>(output_data, off);
                                //float score = abcdk_math_sigmoid(abcdk::torch::util::obj<float>(output_data, off));

                                if (max_score < score)
                                {
                                    max_score = score;
                                    label = c;
                                }
                            }

                            /*低于阈值的不要。*/
                            if (max_score < threshold)
                                continue;

                            abcdk::torch_host::dnn::object one_dst;

                            one_dst.m_label = label;
                            one_dst.m_score = (int)(max_score * 100);

                            size_t x_off = abcdk::torch::util::off<float>(output_packed, output_w, output_w * sizeof(float), output_h, output_c, b, 0, y, 0);
                            size_t y_off = abcdk::torch::util::off<float>(output_packed, output_w, output_w * sizeof(float), output_h, output_c, b, 0, y, 1);
                            size_t w_off = abcdk::torch::util::off<float>(output_packed, output_w, output_w * sizeof(float), output_h, output_c, b, 0, y, 2);
                            size_t h_off = abcdk::torch::util::off<float>(output_packed, output_w, output_w * sizeof(float), output_h, output_c, b, 0, y, 3);

                            float _x = abcdk::torch::util::obj<float>(output_data, x_off); // 矩开中心点X坐标。
                            float _y = abcdk::torch::util::obj<float>(output_data, y_off); // 矩开中心点Y坐标。
                            float _w = abcdk::torch::util::obj<float>(output_data, w_off);
                            float _h = abcdk::torch::util::obj<float>(output_data, h_off);

                            one_dst.m_rect_x1 = (int)(_x - _w / 2);
                            one_dst.m_rect_y1 = (int)(_y - _h / 2);
                            one_dst.m_rect_x2 = (int)(_x + _w / 2);
                            one_dst.m_rect_y2 = (int)(_y + _h / 2);

                            /*修正坐标，不要超出图像范围。*/
                            one_dst.m_rect_x1 = abcdk::torch::util::clamp<int>(one_dst.m_rect_x1, 0, input_w - 1);
                            one_dst.m_rect_y1 = abcdk::torch::util::clamp<int>(one_dst.m_rect_y1, 0, input_h - 1);
                            one_dst.m_rect_x2 = abcdk::torch::util::clamp<int>(one_dst.m_rect_x2, 0, input_w - 1);
                            one_dst.m_rect_y2 = abcdk::torch::util::clamp<int>(one_dst.m_rect_y2, 0, input_h - 1);


                            one_dst.m_keypoint.resize(m_output_kpt_num * 3);

                            for (int c = 4 + output_classes , k = 0; c < output_c; c += output_kpt_size, k += 3)
                            {
                                size_t kpt_x_off = abcdk::torch::util::off<float>(output_packed, output_w, output_w * sizeof(float), output_h, output_c, b, 0, y, c + 0);
                                size_t kpt_y_off = abcdk::torch::util::off<float>(output_packed, output_w, output_w * sizeof(float), output_h, output_c, b, 0, y, c + 1);

                                float _kpt_x = abcdk::torch::util::obj<float>(output_data, kpt_x_off);
                                float _kpt_y = abcdk::torch::util::obj<float>(output_data, kpt_y_off);

                                /*修正坐标，不要超出图像范围。*/
                                one_dst.m_keypoint[k + 0] = abcdk::torch::util::clamp<int>((int)_kpt_x, 0, input_w - 1);
                                one_dst.m_keypoint[k + 1] = abcdk::torch::util::clamp<int>((int)_kpt_y, 0, input_h - 1);
                                one_dst.m_keypoint[k + 2] = 99;
                            }

                            dst[b][label].push_back(one_dst);
                        }
                    }
                }
            };

        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ABCDK_TORCH_HOST_DNN_YOLO_V11_POSE_HXX