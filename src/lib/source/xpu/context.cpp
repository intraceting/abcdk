/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/xpu/context.h"
#include "runtime.in.h"
#include "context.in.h"


static pthread_once_t _abcdk_xpu_context_current_key_status = PTHREAD_ONCE_INIT;
static pthread_key_t _abcdk_xpu_context_current_key = 0xFFFFFFFF;

static void _abcdk_xpu_context_current_key_destroy_cb(void *opaque)
{
    abcdk_xpu_context_t *ctx = (abcdk_xpu_context_t *)opaque;

    ABCDK_TRACE_ASSERT(ctx == NULL, ABCDK_GETTEXT("设备环境内存泄露, 当前线程未解绑设备环境."));
}

static void _abcdk_xpu_context_current_key_create_cb()
{
    pthread_key_create(&_abcdk_xpu_context_current_key, _abcdk_xpu_context_current_key_destroy_cb);
}

static void _abcdk_xpu_context_current_init()
{
    int chk;

    /*only once.*/
    chk = pthread_once(&_abcdk_xpu_context_current_key_status, _abcdk_xpu_context_current_key_create_cb);
    assert(chk == 0);
}

abcdk_xpu_context_t *_abcdk_xpu_context_current_get()
{
    abcdk_xpu_context_t *old_ctx = NULL;

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    old_ctx = (abcdk_xpu_context_t *)pthread_getspecific(_abcdk_xpu_context_current_key);
    
    ABCDK_TRACE_ASSERT(old_ctx != NULL, ABCDK_GETTEXT("当前线程未绑定设备环境."));

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
        return (abcdk_xpu_context_t *)abcdk_xpu::general::context::refer((abcdk_xpu::general::context::metadata_t *)old_ctx);
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return (abcdk_xpu_context_t *)abcdk_xpu::nvidia::context::refer((abcdk_xpu::nvidia::context::metadata_t *)old_ctx);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}

int abcdk_xpu_context_current_set(abcdk_xpu_context_t *ctx)
{
    abcdk_xpu_context_t *old_ctx = NULL;
    abcdk_xpu_context_t *new_ctx = NULL;
    int chk;

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    _abcdk_xpu_context_current_init();
    
    old_ctx = (abcdk_xpu_context_t *)pthread_getspecific(_abcdk_xpu_context_current_key);

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        abcdk_xpu::general::context::unref((abcdk_xpu::general::context::metadata_t **)&old_ctx);
        new_ctx = (abcdk_xpu_context_t *)(ctx ? abcdk_xpu::general::context::refer((abcdk_xpu::general::context::metadata_t *)ctx) : NULL);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        new_ctx = NULL;
#else  // #ifndef HAVE_CUDA
        abcdk_xpu::nvidia::context::unref((abcdk_xpu::nvidia::context::metadata_t **)&old_ctx);
        new_ctx = (abcdk_xpu_context_t *)(ctx ? abcdk_xpu::nvidia::context::refer((abcdk_xpu::nvidia::context::metadata_t *)ctx) : NULL);
#endif // #ifndef HAVE_CUDA
    }

    chk = pthread_setspecific(_abcdk_xpu_context_current_key, new_ctx);
    assert(chk == 0);

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return 0;
}


void abcdk_xpu_context_unref(abcdk_xpu_context_t **ctx)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    _abcdk_xpu_context_current_init();

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
        abcdk_xpu::general::context::unref((abcdk_xpu::general::context::metadata_t **)ctx);
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
#else  // #ifndef HAVE_CUDA
        abcdk_xpu::nvidia::context::unref((abcdk_xpu::nvidia::context::metadata_t **)ctx);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)
}

abcdk_xpu_context_t *abcdk_xpu_context_refer(abcdk_xpu_context_t *ctx)
{
    assert(ctx != NULL);

#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    _abcdk_xpu_context_current_init();

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
        return (abcdk_xpu_context_t *)abcdk_xpu::general::context::refer((abcdk_xpu::general::context::metadata_t *)ctx);
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return (abcdk_xpu_context_t *)abcdk_xpu::nvidia::context::refer((abcdk_xpu::nvidia::context::metadata_t *)ctx);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}

abcdk_xpu_context_t *abcdk_xpu_context_alloc(int id)
{
#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    _abcdk_xpu_context_current_init();

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
        return (abcdk_xpu_context_t *)abcdk_xpu::general::context::alloc(id);
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#ifndef HAVE_CUDA
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #ifndef HAVE_CUDA
        return (abcdk_xpu_context_t *)abcdk_xpu::nvidia::context::alloc(id);
#endif // #ifndef HAVE_CUDA
    }

#endif //#if defined(HAVE_OPENCV) && defined(HAVE_FFMPEG)

    return NULL;
}
