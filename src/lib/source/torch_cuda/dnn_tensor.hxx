/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_DNN_TENSOR_HXX
#define ABCDK_TORCH_NVIDIA_DNN_TENSOR_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/imgutil.h"
#include "abcdk/torch/imgcode.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/opencv.h"

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

namespace abcdk
{
    namespace torch_cuda
    {
        namespace dnn
        {
            class tensor
            {
            public:
                static size_t type_size(nvinfer1::DataType type)
                {
                    size_t size = 0;
    
                    switch (type)
                    {
                    case nvinfer1::DataType::kFLOAT:
                        size = sizeof(float);
                        break;
                    case nvinfer1::DataType::kHALF:
                        size = sizeof(float) / 2;
                        break;
                    case nvinfer1::DataType::kINT8:
                        size = sizeof(int8_t);
                        break;
                    case nvinfer1::DataType::kINT32:
                        size = sizeof(int32_t);
                        break;
                    default:
                        size = 0;
                        break;
                    }
    
                    return size;
                }
                
                static size_t dims_size(nvinfer1::Dims dims, nvinfer1::DataType type)
                {
                    size_t size = type_size(type);
    
                    for (int i = 0; i < dims.nbDims; i++)
                    {
                        size = size * dims.d[i];
                    }
    
                    return size;
                }

            private:
                /*索引(输入输出分别计数）。*/
                int m_index;

                /*名字。*/
                std::string m_name;

                /*模式。*/
                nvinfer1::TensorIOMode m_mode;

                /*类型。*/
                nvinfer1::DataType m_type;

                /*维度。*/
                nvinfer1::Dims m_dims;

                /*数据长度。*/
                size_t m_data_size;

                /*数据对象(host)。*/
                void *m_data_host;

                /*数据对象(cuda)。*/
                void *m_data_cuda;

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
                    m_data_cuda = NULL;
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
                    return (m_mode == nvinfer1::TensorIOMode::kINPUT?1:0);
                }

                nvinfer1::DataType type() const
                {
                    return m_type;
                }

                const nvinfer1::Dims &dims() const
                {
                    return m_dims;
                }

                size_t size() const
                {
                    return m_data_size;
                }

                void *data(int in_host = 0) const
                {
                    return (in_host ? m_data_host : m_data_cuda);
                }

            public:
                void clear()
                {
                    m_index = -1;
                    m_name = "";
                    m_mode = nvinfer1::TensorIOMode::kNONE;
                    m_dims.nbDims = 0;
                    m_dims.d[0] = m_dims.d[1] = m_dims.d[2] = m_dims.d[3] = -1;
                    m_type = (nvinfer1::DataType)-1;

                    m_data_size = 0;
                    
                    abcdk_torch_free_host(&m_data_host);
                    abcdk_torch_free_cuda(&m_data_cuda);

                    for (auto &t : m_input_img_cache)
                        abcdk_torch_image_free_cuda(&t);
                }

                int prepare(int index, const char *name, nvinfer1::TensorIOMode mode, nvinfer1::DataType type, const nvinfer1::Dims &dims, abcdk_option_t *opt)
                {
                    int tensor_b_index, tensor_c_index, tensor_h_index, tensor_w_index;
                    int chk;

                    m_index = index;
                    m_name = (name ? name : "");
                    m_type = type;
                    m_mode = mode;
                    m_dims = dims;

                    ABCDK_ASSERT(m_type == nvinfer1::DataType::kFLOAT, TT("暂时仅支持32位浮点类型。"));

                    if (mode != nvinfer1::TensorIOMode::kINPUT && mode != nvinfer1::TensorIOMode::kOUTPUT)
                        return -1;

                    m_data_size = dims_size(dims, type);

                    m_data_host = abcdk_torch_alloc_z_host(m_data_size);
                    m_data_cuda = abcdk_torch_alloc_z_cuda(m_data_size);

                    if (!m_data_cuda || !m_data_host)
                        return -1;

                    /*输出张量走到这里就结束了。*/
                    if (mode == nvinfer1::TensorIOMode::kOUTPUT)
                        return 0;

                    const char *dims_index_p = abcdk_option_get(opt, "--input-dims-index", 0, "0,1,2,3");

                    const char *img_scale_p = abcdk_option_get(opt, "--input-img-scale", 0, "255,255,255");
                    const char *img_mean_p = abcdk_option_get(opt, "--input-img-mean", 0, "0,0,0");
                    const char *img_std_p = abcdk_option_get(opt, "--input-img-std", 0, "1,1,1");

                    m_input_img_kar = abcdk_option_get_int(opt, "--input-image-keep-aspect-ratio", 0, 0);

                    chk = sscanf(dims_index_p, "%d,%d,%d,%d", &tensor_b_index, &tensor_c_index, &tensor_h_index, &tensor_w_index);
                    if (chk != 4)
                        return -1;

                    m_input_b_size = (tensor_b_index < m_dims.nbDims ? m_dims.d[tensor_b_index] : 1);
                    m_input_c_size = (tensor_c_index < m_dims.nbDims ? m_dims.d[tensor_c_index] : 1);
                    m_input_h_size = (tensor_h_index < m_dims.nbDims ? m_dims.d[tensor_h_index] : 1);
                    m_input_w_size = (tensor_w_index < m_dims.nbDims ? m_dims.d[tensor_w_index] : 1);

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

                    m_input_img_cache.resize(m_input_b_size);

                    return 0;
                }

                int pre_processing(const std::vector<abcdk_torch_image_t *> &img)
                {
                    int dst_dw;
                    int64_t dst_off;
                    float *dst_p;
                    int chk;

                    /*计算步长。*/
                    dst_dw = m_input_w_size * type_size(m_type);

                    for (int i = 0; i < img.size(); i++)
                    {
                        if (i >= m_input_b_size)
                            break;

                        abcdk_torch_image_t *src_img_p = img[i];

                        /*可能未输入图像。*/
                        if (!src_img_p)
                            continue;

                        assert(src_img_p->tag == ABCDK_TORCH_TAG_CUDA);
                        assert(src_img_p->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 || src_img_p->pixfmt == ABCDK_TORCH_PIXFMT_BGR24);

                        abcdk_torch_image_reset_cuda(&m_input_img_cache[i], m_input_w_size, m_input_h_size, src_img_p->pixfmt, 1);
                        if (!src_img_cache_p[i])
                            return -1;

                        abcdk_torch_image_t *src_img_cache_p = m_input_img_cache[i];

                        /*缩放或复制。*/
                        abcdk_torch_imgproc_resize_cuda(src_img_cache_p, NULL, src_img_p, NULL, m_input_img_kar, ABCDK_TORCH_INTER_CUBIC);

                        dst_off = i * m_input_h_size * dst_dw * m_input_c_size;
                        dst_p = ABCDK_PTR2PTR(float,m_data_cuda, dst_off);

                        bool dst_c_invert = false;
                        bool src_c_invert = (src_img_cache_p->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ? true : false);

                        abcdk_torch_imgutil_blob_8u_to_32f_cuda(0, dst_p, dst_dw, dst_c_invert,
                                                                1, src_img_cache_p->data[0], src_img_cache_p->stride[0], src_c_invert,
                                                                1, m_input_w_size, m_input_h_size, m_input_c_size,
                                                                m_input_img_scale, m_input_img_mean, m_input_img_std);

                        // abcdk_torch_imgcode_save_cuda("/tmp/aaa-1.jpg",src_img_cache_p);

                        // abcdk_torch_imgutil_blob_32f_to_8u_cuda(1, src_img_cache_p->data[0], src_img_cache_p->stride[0],
                        //                                         0, dst_p, dst_dw,
                        //                                         1, m_input_w_size, m_input_h_size, m_input_c_size,
                        //                                         m_input_img_scale, m_input_img_mean, m_input_img_std);

                        // abcdk_torch_imgcode_save_cuda("/tmp/aaa-2.jpg",src_img_cache_p);

                        // abcdk_torch_memcpy_cuda(m_data_host,1,m_data_cuda,0,m_data_size);

                        // abcdk_save("/tmp/bbb-1.bin",m_data_host,m_data_size,0);
                        
                        // abcdk_torch_image_t *b = abcdk_torch_image_create_host(m_input_w_size,m_input_h_size,ABCDK_TORCH_PIXFMT_RGB24,1);

                        // abcdk_torch_image_copy_cuda(b,src_img_cache_p);

                        // cv::Mat bb = cv::Mat(m_input_h_size,m_input_w_size,CV_8UC3,(void*)b->data[0],b->stride[0]);
                        // cv::Mat bb2 = cv::dnn::blobFromImage(bb, 1.0 / 255, cv::Size(640, 640), cv::Scalar(0,0,0), false, false);

                        // abcdk_save("/tmp/bbb-2.bin",bb2.data,m_data_size,0);

                        // abcdk_torch_image_free_host(&b);
                    }

                    return 0;
                }

                int post_processing()
                {
                    int chk;

                    chk = abcdk_torch_memcpy_cuda(m_data_host,1,m_data_cuda,0,m_data_size);
                    if(chk != 0)
                        return -1;

                    return 0;
                }
            };

        } // namespace dnn
    } // namespace torch_cuda
} // namespace abcdk

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

#endif // ABCDK_TORCH_NVIDIA_DNN_TENSOR_HXX