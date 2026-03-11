/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_DNN_TENSOR_HXX
#define ABCDK_XPU_NVIDIA_DNN_TENSOR_HXX

#include "abcdk/util/option.h"
#include "abcdk/xpu/types.h"
#include "../base.in.h"
#include "image.hxx"
#include "memory.hxx"
#include "imgproc.hxx"

namespace abcdk_xpu
{

    namespace nvidia
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
                /*索引(输入输出分别计数）.*/
                int m_index;

                /*名字.*/
                std::string m_name;

                /*模式.*/
                nvinfer1::TensorIOMode m_mode;

                /*类型.*/
                nvinfer1::DataType m_type;

                /*维度.*/
                nvinfer1::Dims m_dims;

                /*数据长度.*/
                size_t m_data_size;

                /*数据对象(host).*/
                void *m_data_host;

                /*数据对象(cuda).*/
                void *m_data_cuda;

                /*输入数据维度.*/
                int m_input_b_size;
                int m_input_c_size;
                int m_input_h_size;
                int m_input_w_size;

                /*输入图像系数.*/
                abcdk_xpu_scalar_t m_input_img_scale;

                /*输入图像均值.*/
                abcdk_xpu_scalar_t m_input_img_mean;

                /*输入图像方差.*/
                abcdk_xpu_scalar_t m_input_img_std;

                /*输入图像缓存.*/
                std::vector<image::metadata_t *> m_input_img_cache;

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
                    return (m_mode == nvinfer1::TensorIOMode::kINPUT ? 1 : 0);
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

                    memory::freep(&m_data_host, 1);
                    memory::freep(&m_data_cuda, 0);

                    for (auto &t : m_input_img_cache)
                        image::free(&t);
                }

                int init(int index, const char *name, nvinfer1::TensorIOMode mode, nvinfer1::DataType type, const nvinfer1::Dims &dims)
                {
                    ABCDK_TRACE_ASSERT(type == nvinfer1::DataType::kFLOAT, ABCDK_GETTEXT("暂时仅支持32位浮点类型."));

                    if (mode != nvinfer1::TensorIOMode::kINPUT && mode != nvinfer1::TensorIOMode::kOUTPUT)
                        return -1;

                    m_index = index;
                    m_name = (name ? name : "");
                    m_type = type;
                    m_mode = mode;
                    m_dims = dims;

                    return 0;
                }

                int prepare(abcdk_option_t *opt)
                {
                    int tensor_b_index, tensor_c_index, tensor_h_index, tensor_w_index;
                    int chk;

                    /*计算需要缓存大小.*/
                    m_data_size = dims_size(m_dims, m_type);

                    /*准备缓存.*/
                    m_data_host = memory::alloc_z<void *>(m_data_size, 1);
                    m_data_cuda = memory::alloc_z<void *>(m_data_size, 0);

                    if (!m_data_cuda || !m_data_host)
                        return -1;

                    /*输出张量走到这里就结束了.*/
                    if (m_mode == nvinfer1::TensorIOMode::kOUTPUT)
                        return 0;

                    const char *dims_index_p = abcdk_option_get(opt, "--input-dims-index", 0, "0,1,2,3");

                    const char *img_scale_p = abcdk_option_get(opt, "--input-img-scale", 0, "255,255,255");
                    const char *img_mean_p = abcdk_option_get(opt, "--input-img-mean", 0, "0,0,0");
                    const char *img_std_p = abcdk_option_get(opt, "--input-img-std", 0, "1,1,1");

                    chk = sscanf(dims_index_p, "%d,%d,%d,%d", &tensor_b_index, &tensor_c_index, &tensor_h_index, &tensor_w_index);
                    if (chk != 4)
                        return -1;

                    m_input_b_size = (tensor_b_index < m_dims.nbDims ? m_dims.d[tensor_b_index] : 1);
                    m_input_c_size = (tensor_c_index < m_dims.nbDims ? m_dims.d[tensor_c_index] : 1);
                    m_input_h_size = (tensor_h_index < m_dims.nbDims ? m_dims.d[tensor_h_index] : 1);
                    m_input_w_size = (tensor_w_index < m_dims.nbDims ? m_dims.d[tensor_w_index] : 1);

                    assert(m_input_b_size >= 1 && m_input_c_size == 3 && m_input_h_size > 0 && m_input_w_size > 0);

                    chk = sscanf(img_scale_p, "%f,%f,%f", &m_input_img_scale.f32[0], &m_input_img_scale.f32[1], &m_input_img_scale.f32[2]);
                    if (chk != 3)
                        return -2;

                    chk = sscanf(img_mean_p, "%f,%f,%f", &m_input_img_mean.f32[0], &m_input_img_mean.f32[1], &m_input_img_mean.f32[2]);
                    if (chk != 3)
                        return -3;

                    chk = sscanf(img_std_p, "%f,%f,%f", &m_input_img_std.f32[0], &m_input_img_std.f32[1], &m_input_img_std.f32[2]);
                    if (chk != 3)
                        return -4;

                    m_input_img_cache.resize(m_input_b_size);

                    return 0;
                }

                int pre_processing(const std::vector<image::metadata_t *> &img)
                {
                    int dst_dw;
                    int64_t dst_off;
                    float *dst_p;
                    int chk;

                    /*计算步长.*/
                    dst_dw = m_input_w_size * type_size(m_type);

                    for (int i = 0; i < img.size(); i++)
                    {
                        if (i >= m_input_b_size)
                            break;

                        image::metadata_t *src_img_p = img[i];

                        /*可能未输入图像.*/
                        if (!src_img_p)
                            continue;

                        assert(src_img_p->format == AV_PIX_FMT_RGB24 || src_img_p->format == AV_PIX_FMT_BGR24);

                        image::reset(&m_input_img_cache[i], m_input_w_size, m_input_h_size, pixfmt::ffmpeg_to_local(src_img_p->format), 1, 0);
                        if (!m_input_img_cache[i])
                            return -1;

                        image::metadata_t *src_img_cache_p = m_input_img_cache[i];

                        /*缩放到张量尺寸.*/
                        imgproc::resize(src_img_p, NULL, src_img_cache_p, ABCDK_XPU_INTER_CUBIC);

                        dst_off = i * m_input_h_size * dst_dw * m_input_c_size;
                        dst_p = ABCDK_PTR2PTR(float, m_data_cuda, dst_off);

                        int dst_c_invert = 0;
                        int src_c_invert = (src_img_cache_p->format == AV_PIX_FMT_RGB24 ? 0 : 1);

                        imgproc::blob_8u_to_32f(0, dst_p, dst_dw, dst_c_invert,
                                                1, src_img_cache_p->data[0], src_img_cache_p->linesize[0], src_c_invert,
                                                1, m_input_w_size, m_input_h_size, m_input_c_size,
                                                &m_input_img_scale, &m_input_img_mean, &m_input_img_std);
                    }

                    return 0;
                }

                int post_processing()
                {
                    int chk;

                    chk = memory::copy_1d(m_data_host, 1, m_data_cuda, 0, m_data_size);
                    if (chk != 0)
                        return -1;

                    return 0;
                }
            };

        } // namespace dnn
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_DNN_TENSOR_HXX