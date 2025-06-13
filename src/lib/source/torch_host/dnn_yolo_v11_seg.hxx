/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_YOLO_V11_SEG_HXX
#define ABCDK_TORCH_HOST_DNN_YOLO_V11_SEG_HXX

#include "dnn_model.hxx"

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class yolo_v11_seg : public model
            {
            public:
                yolo_v11_seg(const char *name = "")
                    : model(name)
                {
                }
                virtual ~yolo_v11_seg()
                {
                }

            public:
                virtual void prepare(abcdk_option_t *opt)
                {
                }

            protected:
                virtual void collect_object(std::vector<std::map<int, std::vector<object>>> &dst, std::vector<abcdk_torch_dnn_tensor> &tensor, float threshold)
                {
                    abcdk_torch_dnn_tensor *input_tensor_p = NULL, *output_tensor_p = NULL,*output2_tensor_p = NULL;

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
                            if(t.dims.nb == 3)
                            {
                                ABCDK_ASSERT(output_tensor_p == NULL, TT("YOLO-11模型仅支持一个输出层(nb==3)。"));
                                assert(output_tensor_p == NULL);
                                output_tensor_p = &t;
                            }
                            else if(t.dims.nb == 4)
                            {
                                ABCDK_ASSERT(output2_tensor_p == NULL, TT("YOLO-11模型仅支持一个输出层(nb==4)。"));
                                assert(output2_tensor_p == NULL);
                                output2_tensor_p = &t;
                            }
                        }
                        else
                            continue;
                    }

                    ABCDK_ASSERT(input_tensor_p != NULL, TT("YOLO-11模型至少存在一个输入层。"));
                    ABCDK_ASSERT(output_tensor_p != NULL, TT("YOLO-11模型至少存在一个输出层(nb==3)。"));
                    ABCDK_ASSERT(output2_tensor_p != NULL, TT("YOLO-11模型至少存在一个输出层(nb==4)。"));

                    int input_b, input_c, input_h, input_w;
                    int output_b, output_c, output_h, output_w;
                    int output2_b, output2_c, output2_h, output2_w;

                    input_b = input_tensor_p->dims.d[0];
                    input_c = input_tensor_p->dims.d[1];
                    input_h = input_tensor_p->dims.d[2];
                    input_w = input_tensor_p->dims.d[3];

                    output_b = output_tensor_p->dims.d[0];
                    output_c = output_tensor_p->dims.d[1];
                    output_h = output_tensor_p->dims.d[2];
                    output_w = 1; // output_tensor_p->dims.d[3];

                    output2_b = output2_tensor_p->dims.d[0];
                    output2_c = output2_tensor_p->dims.d[1];
                    output2_h = output2_tensor_p->dims.d[2];
                    output2_w = output2_tensor_p->dims.d[3];

                    assert(input_b == output_b);
                    assert(input_b == output2_b);
                    assert(output_c >= 5);

                    abcdk::torch::ndarray output_data((void *)output_tensor_p->data_p, false, output_b, output_c, output_h, output_w, output_w * sizeof(float));

                    int output_classes = output_c - 4 - output2_c;

                    dst.clear();
                    dst.resize(output_b);

                    for (int b = 0; b < output_b; b++)
                    {
                        for (int y = 0; y < output_h; y++)
                        {
                            int label = -1;
                            float max_score = 0.0;

                            /*cx,cy,w,h,c0,[c1,...]*/

                            /*在所有分类中找出最大的。*/
                            for (int c = 0; c < output_classes ; c++)
                            {
                                float score = output_data.obj<float>(b, 0, y, 4 + c);

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

                            float _x = output_data.obj<float>(b, 0, y, 0); // 矩开中心点X坐标。
                            float _y = output_data.obj<float>(b, 0, y, 1); // 矩开中心点Y坐标。
                            float _w = output_data.obj<float>(b, 0, y, 2);
                            float _h = output_data.obj<float>(b, 0, y, 3);

                            one_dst.m_rect_x1 = (int)(_x - _w / 2);
                            one_dst.m_rect_y1 = (int)(_y - _h / 2);
                            one_dst.m_rect_x2 = (int)(_x + _w / 2);
                            one_dst.m_rect_y2 = (int)(_y + _h / 2);

                            /*修正坐标，不要超出图像范围。*/
                            one_dst.m_rect_x1 = abcdk::torch::util::clamp<int>(one_dst.m_rect_x1, 0, input_w - 1);
                            one_dst.m_rect_y1 = abcdk::torch::util::clamp<int>(one_dst.m_rect_y1, 0, input_h - 1);
                            one_dst.m_rect_x2 = abcdk::torch::util::clamp<int>(one_dst.m_rect_x2, 0, input_w - 1);
                            one_dst.m_rect_y2 = abcdk::torch::util::clamp<int>(one_dst.m_rect_y2, 0, input_h - 1);

#ifdef OPENCV_CORE_HPP

                            /*单个目标的系数。*/
                            cv::Mat mask_coef(1, output2_c, CV_32FC1);

                            for (int m = 0; m < output2_c; m++)
                            {
                                mask_coef.at<float>(0, m) = output_data.obj<float>(b, 0, y, output_classes + 4 + m);
                            }

                            /*完整图像的系数。*/
                            cv::Mat mask_protos = cv::Mat(output2_c, output2_h * output2_w, CV_32FC1,(void*)output2_tensor_p->data_p);

                            /*相乘。*/
                            cv::Mat mask_mul2 = (mask_coef * mask_protos);

                            /*行列转换。*/
                            cv::Mat mask_mul = mask_mul2.t();

                            /*重组维度。*/
                            cv::Mat mask_dest2 = mask_mul.reshape(1, output2_h);

                            /*归一化。*/
                            cv::Mat mask_dest;
                            cv::exp(-mask_dest2, mask_dest);
                            mask_dest = 1.0 / (1.0 + mask_dest);

                            /*还原到输入图像尺寸并复制到节点。*/
                            cv::resize(mask_dest, one_dst.m_segment, cv::Size(input_w, input_h));

#endif //OPENCV_CORE_HPP

                            dst[b][label].push_back(one_dst);
                        }
                    }
                }
            };

        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ABCDK_TORCH_HOST_DNN_YOLO_V11_SEG_HXX