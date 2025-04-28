/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_INFER_LOGGER_HXX
#define ABCDK_TORCH_NVIDIA_INFER_LOGGER_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/nvidia.h"

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

namespace abcdk
{
    namespace torch_cuda
    {
        namespace infer
        {
            class logger : public nvinfer1::ILogger
            {
            private:
                int m_level;

            public:
                logger(int level = LOG_INFO)
                {
                    m_level = level;
                }

                virtual ~logger()
                {
                }

            public:
                virtual void log(nvinfer1::ILogger::Severity level, const char *msg)
                {
                    int qos;

                    if (nvinfer1::ILogger::Severity::kVERBOSE == level)
                        qos = LOG_DEBUG;
                    else if (nvinfer1::ILogger::Severity::kINFO == level)
                        qos = LOG_INFO;
                    else if (nvinfer1::ILogger::Severity::kWARNING == level)
                        qos = LOG_WARNING;
                    else if (nvinfer1::ILogger::Severity::kERROR == level)
                        qos = LOG_ERR;
                    else if (nvinfer1::ILogger::Severity::kINTERNAL_ERROR == level)
                        qos = LOG_ALERT;
                    else
                        qos = LOG_INFO;

                    if (m_level < qos)
                        return;
                    
                    abcdk_trace_printf(qos, "NvLogger(%d): %s", (int)level, msg);
                    
                }
            };

        } // namespace infer
    } // namespace torch_cuda
} // namespace abcdk

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

#endif // ABCDK_TORCH_NVIDIA_INFER_LOGGER_HXX