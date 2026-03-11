/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_DNN_ENGINE_HXX
#define ABCDK_XPU_NVIDIA_DNN_ENGINE_HXX

#include "abcdk/xpu/types.h"
#include "../base.in.h"
#include "../common/util.hxx"
#include "dnn_logger.hxx"
#include "dnn_tensor.hxx"

namespace abcdk_xpu
{
    namespace nvidia
    {
        namespace dnn
        {
            class engine
            {
            private:
                logger m_logger;
                std::vector<tensor> m_tensor_ctx;
                nvinfer1::IExecutionContext *m_exec_ctx;
                nvinfer1::IRuntime *m_runtime_ctx;
                nvinfer1::ICudaEngine *m_engine_ctx;

            public:
                engine()
                {
                    m_exec_ctx = NULL;
                    m_runtime_ctx = NULL;
                    m_engine_ctx = NULL;
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
                    common::util::delete_object(&m_exec_ctx);
                    common::util::delete_object(&m_engine_ctx);
                    common::util::delete_object(&m_runtime_ctx);

                    m_tensor_ctx.clear();
                }

                int load(const void *data, size_t size)
                {
                    assert(data != NULL && size > 0);

                    m_runtime_ctx = nvinfer1::createInferRuntime(m_logger);
                    if (!m_runtime_ctx)
                        return -1;

                    m_engine_ctx = m_runtime_ctx->deserializeCudaEngine(data, size);
                    if (!m_engine_ctx)
                        return -1;

                    m_exec_ctx = m_engine_ctx->createExecutionContext();
                    if (!m_exec_ctx)
                        return -1;

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
                    int index[2] = {0};
                    int chk;

                    m_tensor_ctx.resize(m_engine_ctx->getNbIOTensors());

                    for (int i = 0; i < m_tensor_ctx.size(); i++)
                    {
                        const char *name_p = m_engine_ctx->getIOTensorName(i);
                        nvinfer1::Dims dims = m_engine_ctx->getTensorShape(name_p);
                        nvinfer1::TensorIOMode mode = m_engine_ctx->getTensorIOMode(name_p);
                        nvinfer1::DataType type = m_engine_ctx->getTensorDataType(name_p);
                        int input = (mode == nvinfer1::TensorIOMode::kINPUT);

                        chk = m_tensor_ctx[i].init(index[input ? 0 : 1], name_p, mode, type, dims);
                        if (chk != 0)
                            return -1;

                        chk = m_tensor_ctx[i].prepare(opt);
                        if (chk != 0)
                            return -1;

                        index[input ? 0 : 1]++; // 分别计算输入/输出的索引，索引从0开始为第一个。
                    }

                    return 0;
                }

                int execute(const std::vector<image::metadata_t *> &img)
                {
                    void *bindings[m_tensor_ctx.size()] = {0};
                    bool chk_bool;
                    int chk;

                    for (auto &t : m_tensor_ctx)
                    {
                        if (!t.input())
                            continue;

                        chk = t.pre_processing(img);
                        if (chk != 0)
                            return -1;
                    }

                    for (int i = 0; i < m_tensor_ctx.size(); i++)
                        bindings[i] = m_tensor_ctx[i].data();

                    chk_bool = m_exec_ctx->executeV2(bindings);
                    if (!chk_bool)
                        return -1;

                    for (auto &t : m_tensor_ctx)
                    {
                        if (t.input())
                            continue;

                        chk = t.post_processing();
                        if (chk != 0)
                            return -1;
                    }

                    return 0;
                }
            };
        } // namespace dnn
    } // namespace nvidia
} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_DNN_ENGINE_HXX