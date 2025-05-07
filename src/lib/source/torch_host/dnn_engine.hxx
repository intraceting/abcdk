/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_ENGINE_HXX
#define ABCDK_TORCH_HOST_DNN_ENGINE_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/imgutil.h"
#include "abcdk/torch/onnxruntime.h"
#include "../torch/memory.hxx"
#include "dnn_tensor.hxx"
#include "dnn_logger.hxx"

#ifdef ORT_API_VERSION

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            class engine
            {
            private:
                std::vector<tensor> m_tensor_ctx;
                const OrtApi *m_api_ctx;
                OrtEnv *m_env_ctx;
                OrtSessionOptions *m_ses_opt_ctx;
                OrtSession *m_ses_ctx;
                OrtAllocator *m_alloc_ctx;
                OrtMemoryInfo *m_mem_info;
                OrtValue *m_input_val;
                OrtValue *m_output_val;

            public:
                engine()
                {
                    m_api_ctx = OrtGetApiBase()->GetApi(ORT_API_VERSION);

                    assert(m_api_ctx != NULL);

                    m_env_ctx = NULL;
                    m_ses_opt_ctx = NULL;
                    m_ses_ctx = NULL;
                    m_alloc_ctx = NULL;
                    m_mem_info = NULL;
                    m_input_val = NULL;
                    m_output_val = NULL;
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
                    if (m_ses_ctx)
                        m_api_ctx->ReleaseSession(m_ses_ctx);
                    if (m_ses_opt_ctx)
                        m_api_ctx->ReleaseSessionOptions(m_ses_opt_ctx);
                    if (m_env_ctx)
                        m_api_ctx->ReleaseEnv(m_env_ctx);
                    if (m_mem_info)
                        m_api_ctx->ReleaseMemoryInfo(m_mem_info);
                    if (m_input_val)
                        m_api_ctx->ReleaseValue(m_input_val);
                    if (m_output_val)
                        m_api_ctx->ReleaseValue(m_output_val);

                    m_env_ctx = NULL;
                    m_ses_opt_ctx = NULL;
                    m_ses_ctx = NULL;
                    m_alloc_ctx = NULL;
                    m_mem_info = NULL;
                    m_input_val = NULL;
                    m_output_val = NULL;
                }

                int load(const void *data, size_t size)
                {
                    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
                    dnn::status status(m_api_ctx);

                    assert(data != NULL && size > 0);

                    status = m_api_ctx->GetAllocatorWithDefaultOptions(&m_alloc_ctx);
                    status = m_api_ctx->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m_mem_info);

                    if (!m_alloc_ctx || !m_mem_info)
                        return -2;

                    status = m_api_ctx->CreateEnvWithCustomLogger(dnn::logger, NULL, ORT_LOGGING_LEVEL_VERBOSE, "", &m_env_ctx);
                    status = m_api_ctx->CreateSessionOptions(&m_ses_opt_ctx);

                    if (!m_env_ctx || !m_ses_opt_ctx)
                        return -3;

                    status = m_api_ctx->SetInterOpNumThreads(m_ses_opt_ctx, cpus / 2); // 每个算子内部线程数。
                    status = m_api_ctx->SetInterOpNumThreads(m_ses_opt_ctx, cpus / 4); // 并行执行算子线程数。
                    status = m_api_ctx->SetSessionGraphOptimizationLevel(m_ses_opt_ctx, GraphOptimizationLevel::ORT_ENABLE_ALL);

                    status = m_api_ctx->CreateSessionFromArray(m_env_ctx, data, size, m_ses_opt_ctx, &m_ses_ctx);
                    if (!m_ses_ctx)
                        return -4;

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
                    dnn::status status(m_api_ctx);
                    int k = 0;
                    int chk;

                    size_t input_count = 0, output_count = 0;
                    status = m_api_ctx->SessionGetInputCount(m_ses_ctx, &input_count);
                    status = m_api_ctx->SessionGetOutputCount(m_ses_ctx, &output_count);

                    m_tensor_ctx.resize(input_count + output_count);

                    for (int i = 0; i < input_count; i++)
                    {
                        char *name_p = NULL;
                        status = m_api_ctx->SessionGetInputName(m_ses_ctx, i, m_alloc_ctx, &name_p);

                        OrtTypeInfo *info_p = NULL;
                        status = m_api_ctx->SessionGetInputTypeInfo(m_ses_ctx, i, &info_p);

                        const OrtTensorTypeAndShapeInfo *tensor_info_p = NULL;
                        status = m_api_ctx->CastTypeInfoToTensorInfo(info_p, &tensor_info_p);

                        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                        status = m_api_ctx->GetTensorElementType(tensor_info_p, &type);

                        size_t nd = 0;
                        status = m_api_ctx->GetDimensionsCount(tensor_info_p, &nd);

                        tensor::Shape dims(nd);
                        status = m_api_ctx->GetDimensions(tensor_info_p, dims.data(), nd);

                        chk = m_tensor_ctx[k].prepare(m_api_ctx, m_mem_info, i, name_p, tensor::TensorIOMode::kINPUT, (tensor::DataType)type, dims, opt);

                        m_api_ctx->ReleaseTypeInfo(info_p);                     // free.
                        status = m_api_ctx->AllocatorFree(m_alloc_ctx, name_p); // free.

                        if (chk != 0)
                            return -1;

                        k += 1;
                    }

                    for (int i = 0; i < output_count; i++)
                    {
                        char *name_p = NULL;
                        status = m_api_ctx->SessionGetOutputName(m_ses_ctx, i, m_alloc_ctx, &name_p);

                        OrtTypeInfo *info_p = NULL;
                        status = m_api_ctx->SessionGetOutputTypeInfo(m_ses_ctx, i, &info_p);

                        const OrtTensorTypeAndShapeInfo *tensor_info_p = NULL;
                        status = m_api_ctx->CastTypeInfoToTensorInfo(info_p, &tensor_info_p);

                        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                        status = m_api_ctx->GetTensorElementType(tensor_info_p, &type);

                        size_t nd = 0;
                        status = m_api_ctx->GetDimensionsCount(tensor_info_p, &nd);

                        tensor::Shape dims(nd);
                        status = m_api_ctx->GetDimensions(tensor_info_p, dims.data(), nd);

                        chk = m_tensor_ctx[k].prepare(m_api_ctx, m_mem_info, i, name_p, tensor::TensorIOMode::kOUTPUT, (tensor::DataType)type, dims, opt);

                        m_api_ctx->ReleaseTypeInfo(info_p);                     // free.
                        status = m_api_ctx->AllocatorFree(m_alloc_ctx, name_p); // free.

                        if (chk != 0)
                            return -1;

                        k += 1;
                    }

                    return 0;
                }

                int execute(const std::vector<abcdk_torch_image_t *> &img)
                {
                    int input_count = 0, output_count = 0;
                    char *input_vec_name[m_tensor_ctx.size()] = {0};
                    char *output_vec_name[m_tensor_ctx.size()] = {0};
                    OrtValue *input_vec_value[m_tensor_ctx.size()] = {NULL};
                    OrtValue *output_vec_value[m_tensor_ctx.size()] = {NULL};
                    dnn::status status(m_api_ctx);

                    for (auto &t : m_tensor_ctx)
                    {
                        if (t.input())
                        {
                            t.pre_processing(img);

                            input_vec_name[input_count] = (char *)t.name();
                            input_vec_value[input_count] = (OrtValue *)t.data(0);

                            input_count += 1;
                        }
                        else
                        {
                            output_vec_name[output_count] = (char *)t.name();
                            output_vec_value[output_count] = NULL;

                            output_count += 1;
                        }
                    }

                    status = m_api_ctx->Run(m_ses_ctx, NULL, input_vec_name, input_vec_value, input_count, output_vec_name, output_count, output_vec_value);

                    return 0;
                }
            };
        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ORT_API_VERSION

#endif // ABCDK_TORCH_HOST_DNN_ENGINE_HXX