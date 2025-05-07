/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_HOST_DNN_STATUS_HXX
#define ABCDK_TORCH_HOST_DNN_STATUS_HXX

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
            class status
            {
            private:
                const OrtApi *m_api_ctx;
                OrtStatus *m_status_ptr;

            public:
                status(const OrtApi *api_ctx)
                {
                    m_api_ctx = api_ctx;
                    assert(m_api_ctx != NULL);

                    m_status_ptr = NULL;
                }

                virtual ~status()
                {
                    if (m_status_ptr)
                        m_api_ctx->ReleaseStatus(m_status_ptr);
                }

            public:
                status &operator=(OrtStatus *src)
                {
                    if (m_status_ptr)
                        m_api_ctx->ReleaseStatus(m_status_ptr);

                    m_status_ptr = src;
                    if (m_status_ptr)
                    {
                        const char *msg = m_api_ctx->GetErrorMessage(m_status_ptr);
                        abcdk_trace_printf(LOG_WARNING, "ONNXRuntimeStatus:%s", (msg ? msg : ""));
                    }

                    return *this;
                }

                int check()
                {
                    if (!m_status_ptr)
                        return 0;

                    return -1;
                }
            };

        } // namespace dnn
    } // namespace torch_host
} // namespace abcdk

#endif // ORT_API_VERSION

#endif // ABCDK_TORCH_HOST_DNN_STATUS_HXX