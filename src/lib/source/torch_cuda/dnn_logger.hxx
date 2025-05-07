/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_DNN_LOGGER_HXX
#define ABCDK_TORCH_NVIDIA_DNN_LOGGER_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/nvidia.h"

#if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

namespace abcdk
{
    namespace torch_cuda
    {
        namespace dnn
        {
            class logger : public nvinfer1::ILogger
            {
            public:
                logger()
                {
                }

                virtual ~logger()
                {
                }

            public:
                void log(nvinfer1::ILogger::Severity level, const char *msg) noexcept
                {
                    int type;

                    if (nvinfer1::ILogger::Severity::kVERBOSE == level)
                        type = LOG_DEBUG;
                    else if (nvinfer1::ILogger::Severity::kINFO == level)
                        type = LOG_INFO;
                    else if (nvinfer1::ILogger::Severity::kWARNING == level)
                        type = LOG_WARNING;
                    else if (nvinfer1::ILogger::Severity::kERROR == level)
                        type = LOG_ERR;
                    else if (nvinfer1::ILogger::Severity::kINTERNAL_ERROR == level)
                        type = LOG_ALERT;
                    else
                        type = LOG_INFO;

                    abcdk_trace_printf(type, "NvLogger(%d): %s", (int)level, msg);
                }
            };

        } // namespace dnn
    } // namespace torch_cuda
} // namespace abcdk

#endif // #if defined(__cuda_cuda_h__) && defined(NV_INFER_H)

#endif // ABCDK_TORCH_NVIDIA_DNN_LOGGER_HXX