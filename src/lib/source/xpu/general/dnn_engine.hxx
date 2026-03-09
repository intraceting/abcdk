/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_DNN_ENGINE_HXX
#define ABCDK_XPU_GENERAL_DNN_ENGINE_HXX

#include "abcdk/util/object.h"
#include "abcdk/xpu/types.h"
#include "../runtime.in.h"
#include "../common/util.hxx"
#include "../common/dnn_yolo_v11.hxx"
#include "image.hxx"
#include "dnn_tensor.hxx"


namespace abcdk_xpu
{
    namespace general
    {
        namespace dnn
        {
            class engine
            {
            public:
                static void fetch_tensor_info(onnx::ModelProto &ctx, std::vector<tensor> &tensors, bool in_or_out)
                {
                    auto &graph = ctx.graph();

                    /*真实输入: graph.input − graph.initializer*/
                    auto input_size = [](const onnx::GraphProto &graph)
                    {
                        std::unordered_set<std::string> initializer_names;
                        for (const auto &init : graph.initializer())
                            initializer_names.insert(init.name());

                        size_t count = 0;
                        for (const auto &input : graph.input())
                        {
                            if (initializer_names.count(input.name()) == 0)
                                count += 1;
                        }

                        return count;
                    };

                    int in_count = input_size(graph);
                    int out_count = graph.output_size();

                    if (in_or_out)
                        tensors.resize(in_count + out_count);
                    else
                        assert(tensors.size() == (in_count + out_count));

                    int count = (in_or_out ? in_count : out_count);
                    int index = (in_or_out ? 0 : in_count);

                    for (int i = 0; i < count; i++)
                    {
                        auto &one_tensor = in_or_out ? graph.input(i) : graph.output(i);

                        if (!one_tensor.has_type())
                            continue;

                        if (!one_tensor.type().has_tensor_type())
                            continue;

                        auto &one_tensor_type = one_tensor.type().tensor_type();

                        tensor::data_type type;
                        if (one_tensor_type.elem_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT)
                            type = tensor::data_type::FP32;
                        else if (one_tensor_type.elem_type() == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16)
                            type = tensor::data_type::FP16;
                        else if (one_tensor_type.elem_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT8)
                            type = tensor::data_type::INT8;
                        else if (one_tensor_type.elem_type() == onnx::TensorProto_DataType::TensorProto_DataType_INT32)
                            type = tensor::data_type::INT32;
                        else
                            continue;

                        if (!one_tensor_type.has_shape())
                            continue;

                        auto &one_tensor_shape = one_tensor_type.shape();

                        std::vector<int> dims(one_tensor_shape.dim_size());
                        for (int j = 0; j < one_tensor_shape.dim_size(); ++j)
                        {
                            if (one_tensor_shape.dim(j).has_dim_value())
                                dims[j] = one_tensor_shape.dim(j).dim_value();
                            else
                                dims[j] = -1;
                        }

                        tensors[index++].init(i, one_tensor.name().c_str(), (in_or_out ? tensor::io_mode::INPUT : tensor::io_mode::OUTPUT), type, dims);
                    }

                    if (in_or_out)
                        fetch_tensor_info(ctx, tensors, false);
                }

            private:
                cv::dnn::Net m_ctx;
                std::vector<tensor> m_tensor_ctx;
            public:
                engine()
                {
                }

                virtual ~engine()
                {
                    release();
                }

            public:
                int tensor_size()
                {
                    return m_tensor_ctx.size();
                }

                tensor *tensor_ptr(int index)
                {
                    if (index >= m_tensor_ctx.size())
                        return NULL;

                    return &m_tensor_ctx[index];
                }

            public:
                void release()
                {
                    m_tensor_ctx.clear();
                }

                int load(const void *model_data, size_t model_size)
                {
                    onnx::ModelProto onnx_ctx;
                    bool bchk;

                    bchk = onnx_ctx.ParseFromArray(model_data, model_size);
                    if (!bchk)
                        return -1;

                    m_ctx = cv::dnn::readNetFromONNX((const char *)model_data, model_size);
                    if (m_ctx.empty())
                        return -1;

                    fetch_tensor_info(onnx_ctx, m_tensor_ctx, true);

                    return 0;
                }

                int load(const char *file)
                {
                    abcdk_object_t *model_data;
                    int chk;

                    assert(file != NULL);

                    model_data = abcdk_object_copyfrom_file(file);
                    if (!model_data)
                        return -1;

                    chk = load(model_data->pptrs[0], model_data->sizes[0]);
                    abcdk_object_unref(&model_data);

                    return chk;
                }

                int prepare(abcdk_option_t *opt)
                {
                    int chk;

                    for (int i = 0; i < m_tensor_ctx.size(); i++)
                    {
                        chk = m_tensor_ctx[i].prepare(opt);
                        if (chk != 0)
                            return -1;
                    }

                    return 0;
                }

                int execute(const std::vector<image::metadata_t *> &img)
                {
                    std::string in_name;
                    cv::Mat in_blob;
                    std::vector<cv::Mat> out_blobs;
                    std::vector<std::string> out_names;
                    int chk;

                    for(int i = 0;i<m_tensor_ctx.size();i++)
                    {
                        auto &one = m_tensor_ctx[i];

                        if(one.input())
                        {
                            chk = one.pre_processing(img);
                            if(chk != 0)
                                return -1;

                            in_name = one.name();
                            in_blob = *one.data();
                        }
                        else
                        {
                            out_names.push_back(one.name());
                            out_blobs.push_back(*one.data());
                        }
                        
                    }

                    m_ctx.setInput(in_blob, in_name);
                    m_ctx.forward(out_blobs, out_names);

                    for(int i = 0;i<m_tensor_ctx.size();i++)
                    {
                        auto &one = m_tensor_ctx[i];

                        if(!one.input())
                            continue;

                        one.post_processing();

                    }

                    return 0;
                }
            };
        } // namespace dnn
    } // namespace general
} // namespace abcdk_xpu

#endif // ABCDK_XPU_GENERAL_DNN_ENGINE_HXX