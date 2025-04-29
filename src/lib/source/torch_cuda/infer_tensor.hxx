/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_INFER_TENSOR_HXX
#define ABCDK_TORCH_NVIDIA_INFER_TENSOR_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/tensorproc.h"
#include "abcdk/torch/nvidia.h"

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

namespace abcdk
{
    namespace torch_cuda
    {
        namespace infer
        {
            class tensor
            {
            public:
                static inline size_t type_size(nvinfer1::DataType type)
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

                static inline size_t dims_size(nvinfer1::Dims dims, nvinfer1::DataType type)
                {
                    size_t size = type_size(type);

                    for (int i = 0; i < dims.nbDims; i++)
                    {
                        size = size * dims.d[i];
                    }

                    return size;
                }

            private:
                /*输入层。!0 是，0 否。*/
                int m_input;

                /*索引(输入输出分别计数）。*/
                int m_index;

                /*名字。*/
                std::string m_name;

                /*维度。*/
                nvinfer1::Dims m_dims;

                /*数据类型。*/
                nvinfer1::DataType m_type;

                /*数据对象。*/
                abcdk_torch_tensor_t *m_data;

                /*张量维度尺寸。*/
                int m_b_size;
                int m_c_size;
                int m_h_size;
                int m_w_size;

                /*输入图像等比例缩放。0 否，!0 是。*/
                int m_input_img_kar; // keep aspect ratio

                /*输入图像系数。*/
                float m_input_img_scale[3];

                /*输入图像均值。*/
                float m_input_img_mean[3];

                /*输入图像方差。*/
                float m_input_img_std[3];

                /*输入数据。*/
                abcdk_torch_tensor_t *m_input_data;

                /*输入图像缓存。*/
                std::vector<abcdk_torch_image_t *> m_input_img_cache;

                /*输入图像缩放系数。*/
                std::vector<abcdk_resize_scale_t> m_input_img_resize;

                /*输入图像原始尺寸。*/
                std::vector<abcdk_torch_size_t> m_input_img_size;

            public:
                tensor()
                {
                    m_data = NULL;
                    m_input_data = NULL;
                }

                virtual ~tensor()
                {
                    clear();
                }

            public:
                int input()
                {
                    return m_input;
                }

                int b() const
                {
                    return m_b_size;
                }

                int c() const
                {
                    return m_c_size;
                }

                int h() const
                {
                    return m_h_size;
                }

                int w() const
                {
                    return m_w_size;
                }

                void *data() const
                {
                    return m_data->data;
                }

                void copy_to(int dst_in_host, abcdk_torch_tensor_t **dst)
                {
                    if (dst_in_host)
                        abcdk_torch_tensor_reset_host(dst, m_data->format, m_data->block, m_data->width, m_data->height, m_data->depth, m_data->cell, 1);
                    else
                        abcdk_torch_tensor_reset_cuda(dst, m_data->format, m_data->block, m_data->width, m_data->height, m_data->depth, m_data->cell, 1);

                    abcdk_torch_tensor_copy_cuda(m_data, *dst);
                }

            public:
                void clear()
                {
                    abcdk_torch_tensor_free_cuda(&m_data);
                    abcdk_torch_tensor_free_cuda(&m_input_data);

                    for (auto &t : m_input_img_cache)
                        abcdk_torch_image_free_cuda(&t);

                    m_input_img_resize.clear();
                    m_input_img_size.clear();

                    m_input = 0;
                    m_index = -1;
                    m_name = "";
                    m_dims.nbDims = 0;
                    m_dims.d[0] = m_dims.d[1] = m_dims.d[2] = m_dims.d[3] = -1;
                    m_type = (nvinfer1::DataType)-1;
                }

                int prepare(int input, int index, const char *name, nvinfer1::DataType type, const nvinfer1::Dims &dims, abcdk_option_t *opt)
                {
                    int tensor_b_index, tensor_c_index, tensor_h_index, tensor_w_index;
                    int tensor_cell_size;
                    int chk;

                    m_input = (input ? 1 : 0);
                    m_index = index;
                    m_name = (name ? name : "");
                    m_dims = dims;
                    m_type = type;

                    ABCDK_ASSERT(m_type == nvinfer1::DataType::kFLOAT, TT("暂时仅支持32位浮点类型。"));

                    tensor_cell_size = type_size(type);

                    if (!m_input)
                    {
                        m_b_size = (m_dims.nbDims > 0 ? m_dims.d[0] : 1);
                        m_c_size = (m_dims.nbDims > 1 ? m_dims.d[1] : 1);
                        m_h_size = (m_dims.nbDims > 2 ? m_dims.d[2] : 1);
                        m_w_size = (m_dims.nbDims > 3 ? m_dims.d[3] : 1);

                        abcdk_torch_tensor_reset_cuda(&m_data, ABCDK_TORCH_TENFMT_NCHW, m_b_size, m_w_size, m_h_size, m_c_size, tensor_cell_size, 1);

                        if (!m_data)
                            return -1;
                    }
                    else
                    {
                        const char *tensor_dims_p = abcdk_option_get(opt, "--input-tensor-dims-index", 0, "0,1,2,3");

                        const char *img_scale_p = abcdk_option_get(opt, "--input-img-scale", 0, "255,255,255");
                        const char *img_mean_p = abcdk_option_get(opt, "--input-img-mean", 0, "0,0,0");
                        const char *img_std_p = abcdk_option_get(opt, "--input-img-std", 0, "1,1,1");

                        m_input_img_kar = abcdk_option_get_int(opt, "--input-image-keep-aspect-ratio", 0, 0);

                        chk = sscanf(tensor_dims_p, "%d,%d,%d,%d", &tensor_b_index, &tensor_c_index, &tensor_h_index, &tensor_w_index);
                        if (chk != 4)
                            return -5;

                        m_b_size = (tensor_b_index < m_dims.nbDims ? m_dims.d[tensor_b_index] : 1);
                        m_c_size = (tensor_c_index < m_dims.nbDims ? m_dims.d[tensor_c_index] : 1);
                        m_h_size = (tensor_h_index < m_dims.nbDims ? m_dims.d[tensor_h_index] : 1);
                        m_w_size = (tensor_w_index < m_dims.nbDims ? m_dims.d[tensor_w_index] : 1);

                        assert(m_b_size >= 1 && m_c_size == 3 && m_h_size > 0 && m_w_size > 0);

                        abcdk_torch_tensor_reset_cuda(&m_data, ABCDK_TORCH_TENFMT_NCHW, m_b_size, m_w_size, m_h_size, m_c_size, tensor_cell_size, 1);

                        abcdk_torch_tensor_reset_cuda(&m_input_data, ABCDK_TORCH_TENFMT_NHWC, m_b_size, m_w_size, m_h_size, m_c_size, 1, 1);

                        if (!m_data || !m_input_data)
                            return -1;

                        chk = sscanf(img_scale_p, "%f,%f,%f", &m_input_img_scale[0], &m_input_img_scale[1], &m_input_img_scale[2]);
                        if (chk != 3)
                            return -2;

                        chk = sscanf(img_mean_p, "%f,%f,%f", &m_input_img_mean[0], &m_input_img_mean[1], &m_input_img_mean[2]);
                        if (chk != 3)
                            return -3;

                        chk = sscanf(img_std_p, "%f,%f,%f", &m_input_img_std[0], &m_input_img_std[1], &m_input_img_std[2]);
                        if (chk != 3)
                            return -4;

                        m_input_img_cache.resize(m_b_size);
                        m_input_img_resize.resize(m_b_size);
                        m_input_img_size.resize(m_b_size);

                        for (int i = 0; i < m_b_size; i++)
                        {
                            abcdk_torch_image_reset_cuda(&m_input_img_cache[i], m_w_size, m_h_size, ABCDK_TORCH_PIXFMT_RGB24, 1);
                            if (!m_input_img_cache[i])
                                return -6;
                        }
                    }

                    return 0;
                }

                int pretreatment(const std::vector<abcdk_torch_image_t *> &img)
                {
                    int chk;

                    for (int i = 0; i < img.size(); i++)
                    {
                        if (i >= m_b_size)
                            break;

                        abcdk_torch_image_t *src_img_p = img[i];
                        abcdk_torch_image_t *dst_img_p = m_input_img_cache[i];
                        abcdk_resize_scale_t *img_resize_p = &m_input_img_resize[i];
                        abcdk_torch_size_t *img_size_p = &m_input_img_size[i];

                        /*可能未输入图像。*/
                        if (!src_img_p)
                            continue;

                        assert(src_img_p->tag == ABCDK_TORCH_TAG_CUDA);
                        assert(src_img_p->pixfmt == ABCDK_TORCH_PIXFMT_RGB24);

                        abcdk_resize_ratio_2d(img_resize_p, src_img_p->width, src_img_p->height, m_w_size, m_h_size, m_input_img_kar);

                        /*缩放或复制。*/
                        abcdk_torch_imgproc_resize_cuda(dst_img_p, NULL, src_img_p, NULL, m_input_img_kar, NPPI_INTER_CUBIC);

                        abcdk_torch_tensor_copy_block_cuda(m_input_data, i, dst_img_p->data[0], dst_img_p->stride[0]);
                    }

                    abcdk_torch_tensorproc_blob_8u_to_32f_cuda(m_data, m_input_data, m_input_img_scale, m_input_img_mean, m_input_img_std);

                    return 0;
                }
            };

        } // namespace infer
    } // namespace torch_cuda
} // namespace abcdk

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

#endif // ABCDK_TORCH_NVIDIA_INFER_TENSOR_HXX