/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_TENSOR_HXX
#define ABCDK_TORCH_HOST_DNN_TENSOR_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/imgutil.h"
#include "abcdk/torch/imgcode.h"
#include "abcdk/torch/opencv.h"

#ifdef OPENCV_DNN_HPP

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class tensor
            {
            public:
                enum DataType
                {
                    //! 32-bit floating point format.
                    kFLOAT = CV_32FC1,

                    //! IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand.
                    kHALF = CV_16FC1,

                    //! Signed 8-bit integer representing a quantized floating-point value.
                    kINT8 = CV_8SC1,

                    //! Signed 32-bit integer format.
                    kINT32 = CV_32SC1
                };
                
                enum TensorIOMode
                {
                    //! Tensor is not an input or output.
                    kNONE = 0,

                    //! Tensor is input to the engine.
                    kINPUT = 1,

                    //! Tensor is output by the engine.
                    kOUTPUT = 2
                };

                static size_t type_size(DataType type)
                {
                    size_t size = 0;
    
                    switch (type)
                    {
                    case DataType::kFLOAT:
                        size = sizeof(float);
                        break;
                    case DataType::kHALF:
                        size = sizeof(float) / 2;
                        break;
                    case DataType::kINT8:
                        size = sizeof(int8_t);
                        break;
                    case DataType::kINT32:
                        size = sizeof(int32_t);
                        break;
                    default:
                        size = 0;
                        break;
                    }
    
                    return size;
                }
                
                static size_t dims_size(const cv::dnn::MatShape &dims, DataType type)
                {
                    size_t size = type_size(type);
    
                    for (int i = 0; i < dims.size(); i++)
                    {
                        size = size * dims[i];
                    }
    
                    return size;
                }


            private:
                /*索引(输入输出分别计数）。*/
                int m_index;

                /*名字。*/
                std::string m_name;

                /*模式。*/
                TensorIOMode m_mode;

                /*类型。*/
                DataType m_type;

                /*维度。*/
                cv::dnn::MatShape m_dims;

                /*数据长度。*/
                size_t m_data_size;

                /*数据对象(host)。*/
                void *m_data_host;

                /*输入数据维度。*/
                int m_input_b_size;
                int m_input_c_size;
                int m_input_h_size;
                int m_input_w_size;

                /*输入图像等比例缩放。0 否，!0 是。*/
                int m_input_img_kar; // keep aspect ratio

                /*输入图像系数。*/
                float m_input_img_scale[3];

                /*输入图像均值。*/
                float m_input_img_mean[3];

                /*输入图像方差。*/
                float m_input_img_std[3];

                /*输入图像缓存。*/
                std::vector<abcdk_torch_image_t *> m_input_img_cache;

            public:
                tensor()
                {
                    m_data_host = NULL;
                }

                virtual ~tensor()
                {
                    clear();
                }

            public:
                int index() const
                {
                    return m_index;
                }

                const char *name() const 
                {
                    return m_name.c_str();
                }

                int input() const
                {
                    return (m_mode == TensorIOMode::kINPUT?1:0);
                }

                DataType type() const
                {
                    return m_type;
                }

                const cv::dnn::MatShape &dims() const
                {
                    return m_dims;
                }

                size_t size() const
                {
                    return m_data_size;
                }

                void *data() const
                {
                    return m_data_host;
                }

            public:
                void clear()
                {
                    m_index = -1;
                    m_name = "";
                    m_mode = TensorIOMode::kNONE;
                    m_dims.clear();
                    m_type = DataType::kFLOAT;

                    m_data_size = 0;
                    
                    abcdk_torch_free_host(&m_data_host);

                    for (auto &t : m_input_img_cache)
                        abcdk_torch_image_free_host(&t);
                }

                int prepare(int index, const char *name, TensorIOMode mode, DataType type, const cv::dnn::MatShape &dims, abcdk_option_t *opt)
                {
                    int tensor_b_index, tensor_c_index, tensor_h_index, tensor_w_index;
                    int chk;

                    m_index = index;
                    m_name = (name ? name : "");
                    m_type = type;
                    m_mode = mode;
                    m_dims = dims;

                    ABCDK_ASSERT(m_type == DataType::kFLOAT, TT("暂时仅支持32位浮点类型。"));

                    if (mode != TensorIOMode::kINPUT && mode != TensorIOMode::kOUTPUT)
                        return -1;

                    m_data_size = dims_size(dims, type);

                    m_data_host = abcdk_torch_alloc_z_host(m_data_size);

                    if (!m_data_host)
                        return -1;

                    /*输出张量走到这里就结束了。*/
                    if (mode == TensorIOMode::kOUTPUT)
                        return 0;

                    const char *dims_index_p = abcdk_option_get(opt, "--input-dims-index", 0, "0,1,2,3");

                    const char *img_scale_p = abcdk_option_get(opt, "--input-img-scale", 0, "255,255,255");
                    const char *img_mean_p = abcdk_option_get(opt, "--input-img-mean", 0, "0,0,0");
                    const char *img_std_p = abcdk_option_get(opt, "--input-img-std", 0, "1,1,1");

                    m_input_img_kar = abcdk_option_get_int(opt, "--input-image-keep-aspect-ratio", 0, 0);

                    chk = sscanf(dims_index_p, "%d,%d,%d,%d", &tensor_b_index, &tensor_c_index, &tensor_h_index, &tensor_w_index);
                    if (chk != 4)
                        return -1;

                    m_input_b_size = (tensor_b_index < m_dims.size() ? m_dims[tensor_b_index] : 1);
                    m_input_c_size = (tensor_c_index < m_dims.size() ? m_dims[tensor_c_index] : 1);
                    m_input_h_size = (tensor_h_index < m_dims.size() ? m_dims[tensor_h_index] : 1);
                    m_input_w_size = (tensor_w_index < m_dims.size() ? m_dims[tensor_w_index] : 1);

                    assert(m_input_b_size >= 1 && m_input_c_size == 3 && m_input_h_size > 0 && m_input_w_size > 0);

                    chk = sscanf(img_scale_p, "%f,%f,%f", &m_input_img_scale[0], &m_input_img_scale[1], &m_input_img_scale[2]);
                    if (chk != 3)
                        return -2;

                    chk = sscanf(img_mean_p, "%f,%f,%f", &m_input_img_mean[0], &m_input_img_mean[1], &m_input_img_mean[2]);
                    if (chk != 3)
                        return -3;

                    chk = sscanf(img_std_p, "%f,%f,%f", &m_input_img_std[0], &m_input_img_std[1], &m_input_img_std[2]);
                    if (chk != 3)
                        return -4;

                    return 0;
                }

                int pre_processing(const std::vector<abcdk_torch_image_t *> &img)
                {
                    int dst_dw;
                    int64_t dst_off;
                    float *dst_p;
                    int chk;

                    /*图像缓存创建一次即可。*/
                    if (m_input_img_cache.size() <= 0)
                    {
                        m_input_img_cache.resize(m_input_b_size);
                        for (int i = 0; i < m_input_b_size; i++)
                        {
                            abcdk_torch_image_reset_cuda(&m_input_img_cache[i], m_input_w_size, m_input_h_size, ABCDK_TORCH_PIXFMT_RGB24, 1);
                            if (!m_input_img_cache[i])
                                return -1;
                        }
                    }

                    /*计算步长。*/
                    dst_dw = m_input_w_size * type_size(m_type);

                    for (int i = 0; i < img.size(); i++)
                    {
                        if (i >= m_input_b_size)
                            break;

                        abcdk_torch_image_t *src_img_p = img[i];
                        abcdk_torch_image_t *src_img_cache_p = m_input_img_cache[i];

                        /*可能未输入图像。*/
                        if (!src_img_p)
                            continue;

                        assert(src_img_p->tag == ABCDK_TORCH_TAG_HOST);
                        assert(src_img_p->pixfmt == ABCDK_TORCH_PIXFMT_RGB24);
                        assert(src_img_cache_p->pixfmt == ABCDK_TORCH_PIXFMT_RGB24);

                        /*缩放或复制。*/
                        abcdk_torch_imgproc_resize_host(src_img_cache_p, NULL, src_img_p, NULL, m_input_img_kar, cv::INTER_CUBIC);
                        

                        dst_off = i * m_input_h_size * dst_dw * m_input_c_size;
                        dst_p = ABCDK_PTR2PTR(float,m_data_host, dst_off);

                        abcdk_torch_imgutil_blob_8u_to_32f_host(0, dst_p, dst_dw,
                                                                1, src_img_cache_p->data[0], src_img_cache_p->stride[0],
                                                                1, m_input_w_size, m_input_h_size, m_input_c_size,
                                                                m_input_img_scale, m_input_img_mean, m_input_img_std);
                        }

                    return 0;
                }

                int post_processing()
                {
                    return 0;
                }
            };

        } // namespace dnn
    } // namespace torch_cuda
} // namespace abcdk

#endif // OPENCV_DNN_HPP

#endif // ABCDK_TORCH_HOST_DNN_TENSOR_HXX