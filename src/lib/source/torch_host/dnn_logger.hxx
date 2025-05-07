/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_LOGGER_HXX
#define ABCDK_TORCH_HOST_DNN_LOGGER_HXX

#include "abcdk/util/option.h"
#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/imgutil.h"
#include "abcdk/torch/onnxruntime.h"
#include "../torch/memory.hxx"
#include "dnn_tensor.hxx"


#ifdef ORT_API_VERSION

namespace abcdk
{
    namespace torch_host
    {
        namespace dnn
        {
            static inline void logger(void *param, OrtLoggingLevel level, const char *category, const char *logid, const char *code_location, const char *message)
            {
                int type;

                if (ORT_LOGGING_LEVEL_VERBOSE == level)
                    type = LOG_DEBUG;
                else if (ORT_LOGGING_LEVEL_INFO == level)
                    type = LOG_INFO;
                else if (ORT_LOGGING_LEVEL_WARNING == level)
                    type = LOG_WARNING;
                else if (ORT_LOGGING_LEVEL_ERROR == level)
                    type = LOG_ERR;
                else if (ORT_LOGGING_LEVEL_FATAL == level)
                    type = LOG_ALERT;
                else
                    type = LOG_INFO;

                abcdk_trace_printf(type, "ONNXRuntimeLogger(%d): %s", (int)level, message);
            }

        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ORT_API_VERSION

#endif // ABCDK_TORCH_HOST_DNN_LOGGER_HXX