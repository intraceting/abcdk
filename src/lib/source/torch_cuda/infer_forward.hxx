/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_INFER_FORWARD_HXX
#define ABCDK_TORCH_NVIDIA_INFER_FORWARD_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/tensorproc.h"
#include "abcdk/torch/nvidia.h"
#include "../torch/memory.hxx"
#include "infer_logger.hxx"

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

namespace abcdk
{
    namespace torch_cuda
    {
        namespace infer
        {
            class forward
            {
            private:
                logger m_logger;

                nvinfer1::IBuilder *m_builder_ctx;
                nvinfer1::INetworkDefinition *m_network_ctx;
                nvinfer1::IBuilderConfig *m_config_ctx;
                nvinfer1::ICudaEngine *m_engine_ctx;
                nvinfer1::IHostMemory *m_stream_ctx;
                nvonnxparser::IParser *m_onnx_ctx;

            public:
                forward()
                {
                    m_builder_ctx = NULL;
                    m_network_ctx = NULL;
                    m_config_ctx = NULL;
                    m_engine_ctx = NULL;
                    m_stream_ctx = NULL;
                    m_onnx_ctx = NULL;
                }

                virtual ~forward()
                {
                    destroy();
                }

            public:
                logger *logger_ctx()
                {
                    return &m_logger;
                }

                void dump()
                {
                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL && m_onnx_ctx != NULL);

                    for (int i = 0; i < m_network_ctx->getNbInputs(); ++i)
                    {
                        abcdk_trace_printf(LOG_INFO, "InputTensorName[i]: %s",i , m_network_ctx->getInput(i)->getName());
                    }

                    for (int i = 0; i < m_network_ctx->getNbLayers(); ++i)
                    {
                        for (int j = 0; j < m_network_ctx->getLayer(i)->getNbOutputs(); ++j)
                        {
                            abcdk_trace_printf(LOG_INFO, "OutputTensorName[%d][%d]: %s",i,j, m_network_ctx->getLayer(i)->getOutput(j)->getName());
                        }
                    }
                }

                int save(const char *file)
                {
                    int chk;

                    assert(m_engine_ctx != NULL && m_stream_ctx != NULL);

                    assert(file != NULL);

                    if(access(file,F_OK) == 0)
                    {
                        chk = truncate(file, 0);
                        if (chk != 0)
                            return -1;
                    }

                    chk = abcdk_save(file, m_stream_ctx->data(), m_stream_ctx->size(),0);
                    if(chk != m_stream_ctx->size())
                        return -2;

                    return 0;

                }

                void destroy()
                {
                    abcdk::torch::memory::delete_object(&m_onnx_ctx);
                    abcdk::torch::memory::delete_object(&m_engine_ctx);
                    abcdk::torch::memory::delete_object(&m_stream_ctx);
                    abcdk::torch::memory::delete_object(&m_config_ctx);
                    abcdk::torch::memory::delete_object(&m_network_ctx);
                    abcdk::torch::memory::delete_object(&m_builder_ctx);
                }

                int create(uint32_t flag)
                {

                    m_builder_ctx = nvinfer1::createInferBuilder(m_logger);
                    if (!m_builder_ctx)
                        return -1;

                    m_network_ctx = m_builder_ctx->createNetworkV2(flag);
                    if (!m_network_ctx)
                        return -2;

                    m_config_ctx = m_builder_ctx->createBuilderConfig();
                    if (!m_config_ctx)
                        return -3;

                    return 0;
                }

                void enable_dla(int core = 0)
                {
                    int core_nb;

                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL);

                    core_nb = m_builder_ctx->getNbDLACores();

                    assert(core_nb > core);

                    m_config_ctx->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
                    m_config_ctx->setDLACore(core);

                    for (int i = 0; i < m_network_ctx->getNbLayers(); i++)
                    {
                        nvinfer1::ILayer *layer_p = m_network_ctx->getLayer(i);
                        if (m_config_ctx->canRunOnDLA(layer_p))
                        {
                            abcdk_trace_printf(LOG_INFO, "Set layer '%s' to run on DLA.", layer_p->getName());

                            m_config_ctx->setDeviceType(layer_p, nvinfer1::DeviceType::kDLA);
                        }
                        else
                        {
                            abcdk_trace_printf(LOG_INFO, "The layer '%s' does not support DLA.", layer_p->getName());
                        }
                    }
                }

                int build()
                {
                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL);

                    abcdk::torch::memory::delete_object(&m_engine_ctx);
                    abcdk::torch::memory::delete_object(&m_stream_ctx);

                    m_engine_ctx = m_builder_ctx->buildEngineWithConfig(*m_network_ctx, *m_config_ctx);
                    if (!m_engine_ctx)
                        return -1;

                    m_stream_ctx = m_builder_ctx->buildSerializedNetwork(*m_network_ctx, *m_config_ctx);
                    if (!m_stream_ctx)
                        return -1;

                    return 0;
                }

                int load_onnx(const char *file)
                {
                    bool chk_bool;

                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL);

                    assert(file != NULL);

                    abcdk::torch::memory::delete_object(&m_onnx_ctx);

                    m_onnx_ctx = nvonnxparser::createParser(*m_network_ctx, m_logger);
                    if (!m_onnx_ctx)
                        return -1;

                    chk_bool = m_onnx_ctx->parseFromFile(file, (int)nvinfer1::ILogger::Severity::kVERBOSE);
                    if (!chk_bool)
                        return -2;

                    return 0;
                }

                int get_input_nb()
                {
                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL && m_onnx_ctx != NULL);

                    return m_network_ctx->getNbInputs();
                }

                int get_input_dims(int bchw[4], const int idx[4], int layer = 0)
                {
                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL && m_onnx_ctx != NULL);

                    assert(bchw != NULL && idx != NULL && layer >= 0);

                    nvinfer1::ITensor *tensor_p = m_network_ctx->getInput(layer);
                    if (!tensor_p)
                        return -1;

                    nvinfer1::Dims dims = tensor_p->getDimensions();

                    for (int i = 0; i < 4; i++)
                    {
                        if (idx[i] >= 0 && dims.nbDims > idx[i])
                            bchw[i] = dims.d[idx[i]];
                    }

                    return 0;
                }

                int set_input_dims(int bchw[4], const int idx[4], int layer = 0)
                {
                    nvinfer1::Dims4 dims;

                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL && m_onnx_ctx != NULL);

                    assert(bchw != NULL && idx != NULL && layer >= 0);

                    for (int i = 0; i < 4; i++)
                    {
                        if (idx[i] >= 0 && dims.nbDims > idx[i])
                            dims.d[idx[i]] = bchw[i];
                    }

                    nvinfer1::ITensor *tensor_p = m_network_ctx->getInput(layer);
                    if (!tensor_p)
                        return -1;

                    tensor_p->setDimensions(dims);

                    return 0;
                }

                void set_fp16()
                {
                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL);

                    if (m_builder_ctx->platformHasFastFp16())
                    {
                        m_config_ctx->setFlag(nvinfer1::BuilderFlag::kFP16);
                    }
                    else
                    {
                        abcdk_trace_printf(LOG_INFO, TT("不支持FP16，忽略并使用默认类型。"));
                    }
                }

                void set_int8()
                {
                    assert(m_builder_ctx != NULL && m_config_ctx != NULL && m_network_ctx != NULL);

                    if (m_builder_ctx->platformHasFastInt8())
                    {
                        m_config_ctx->setFlag(nvinfer1::BuilderFlag::kINT8);
                    }
                    else
                    {
                        abcdk_trace_printf(LOG_INFO, TT("不支持int8，忽略并使用默认类型。"));
                    }
                }
            };

        } // namespace infer
    } // namespace torch_cuda
} // namespace abcdk

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

#endif // ABCDK_TORCH_NVIDIA_INFER_FORWARD_HXX