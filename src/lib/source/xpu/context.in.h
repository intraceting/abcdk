/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_CONTEXT_IN_H
#define ABCDK_XPU_CONTEXT_IN_H

#include "abcdk/xpu/context.h"
#include "runtime.in.h"

#if defined(__XPU_GENERAL__)
#include "general/context.hxx"
#if defined(__XPU_NVIDIA__)
#include "nvidia/context.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

namespace abcdk_xpu
{
    namespace context
    {
        class guard
        {
        private:
            abcdk_xpu_context_t *m_ctx;

        public:
            guard(abcdk_xpu_context_t *ctx)
            {
#if defined(__XPU_GENERAL__)

                if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
                {
                    int chk = abcdk_xpu::general::context::current_push((abcdk_xpu::general::context::metadata_t *)(m_ctx = ctx));
                    assert(chk == 0);
                }
                else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
                {
#if !defined(__XPU_NVIDIA__)
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
                    assert(0);
#else  // #if !defined(__XPU_NVIDIA__)
                    int chk = abcdk_xpu::nvidia::context::current_push((abcdk_xpu::nvidia::context::metadata_t *)(m_ctx = ctx));
                    assert(chk == 0);
#endif // #if !defined(__XPU_NVIDIA__)
                }

#endif //#if defined(__XPU_GENERAL__)
            }

            virtual ~guard()
            {
#if defined(__XPU_GENERAL__)

                if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
                {
                    int chk = abcdk_xpu::general::context::current_pop((abcdk_xpu::general::context::metadata_t *)m_ctx);
                    assert(chk == 0);
                    abcdk_xpu::general::context::unref((abcdk_xpu::general::context::metadata_t **)&m_ctx);
                }
                else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
                {
#if !defined(__XPU_NVIDIA__)
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
                    assert(0);
#else  // #if !defined(__XPU_NVIDIA__)
                    int chk = abcdk_xpu::nvidia::context::current_pop((abcdk_xpu::nvidia::context::metadata_t *)m_ctx);
                    assert(chk == 0);
                    abcdk_xpu::nvidia::context::unref((abcdk_xpu::nvidia::context::metadata_t **)&m_ctx);
#endif // #if !defined(__XPU_NVIDIA__)
                }

#endif //#if defined(__XPU_GENERAL__)
            }
        };

    } // namespace context
} // namespace abcdk_xpu

abcdk_xpu_context_t *_abcdk_xpu_context_current_get();

#endif // ABCDK_XPU_CONTEXT_IN_H