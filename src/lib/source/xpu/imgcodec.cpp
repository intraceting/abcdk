/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/io.h"
#include "abcdk/curl/util.h"
#include "abcdk/xpu/imgcodec.h"
#include "runtime.in.h"
#include "context.in.h"

#if defined(__XPU_GENERAL__)
#include "general/imgcodec.hxx"
#if defined(__XPU_NVIDIA__)
#include "nvidia/imgcodec.hxx"
#endif //#if defined(__XPU_NVIDIA__)
#endif //#if defined(__XPU_GENERAL__)

abcdk_object_t *abcdk_xpu_imgcodec_encode(const abcdk_xpu_image_t *src, const char *ext)
{
    assert(src != NULL);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return abcdk_xpu::general::imgcodec::encode((abcdk_xpu::general::image::metadata_t *)src, ext);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #if !defined(__XPU_NVIDIA__)
        return abcdk_xpu::nvidia::imgcodec::encode((abcdk_xpu::nvidia::image::metadata_t *)src, ext);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return NULL;
}

int abcdk_xpu_imgcodec_encode_to_file(const abcdk_xpu_image_t *src, const char *dst, const char *ext)
{
    abcdk_object_t *dst_data = NULL;
    ssize_t wr_size;
    int chk;

    assert(src != NULL && dst != NULL);

    if(!ext)
        ext = strrchr(dst,'.');//提取扩展名.

    dst_data = abcdk_xpu_imgcodec_encode(src, ext);
    if (!dst_data)
        return -1;

    wr_size = abcdk_dump(dst, dst_data->pptrs[0], dst_data->sizes[0]);
    chk = (wr_size == dst_data->sizes[0] ? 0 : -1);
    abcdk_object_unref(&dst_data);

    return chk;
}

abcdk_xpu_image_t *abcdk_xpu_imgcodec_decode(const void *src, size_t size)
{
    assert(src != NULL && size > 0);

#if defined(__XPU_GENERAL__)

    abcdk_xpu::context::guard context_push_pop(_abcdk_xpu_context_current_get());

    if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NONE)
    {
        return (abcdk_xpu_image_t *)abcdk_xpu::general::imgcodec::decode(src, size);
    }
    else if (_abcdk_xpu_hwaccel_get() == ABCDK_XPU_HWACCEL_NVIDIA)
    {
#if !defined(__XPU_NVIDIA__)
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含NVIDIA工具."));
        return NULL;
#else  // #if !defined(__XPU_NVIDIA__)
        return (abcdk_xpu_image_t *)abcdk_xpu::nvidia::imgcodec::decode(src, size);
#endif // #if !defined(__XPU_NVIDIA__)
    }

#endif //#if defined(__XPU_GENERAL__)

    return NULL;
}

abcdk_xpu_image_t *abcdk_xpu_imgcodec_decode_from_file(const char *src)
{
    char tmpfile[] = {"/tmp/abcdk-XXXXXX"};
    int tmp_fd = -1;
    abcdk_object_t *tmp_data = NULL;
    abcdk_xpu_image_t *img;
    int chk;

    assert(src != NULL);

    if (abcdk_strncmp(src, "https://", 8, 0) == 0 ||
        abcdk_strncmp(src, "http://", 7, 0) == 0 ||
        abcdk_strncmp(src, "ftp://", 6, 0) == 0)
    {
        tmp_fd = mkstemp(tmpfile);
        if (tmp_fd < 0)
            return NULL;

        chk = abcdk_curl_download_fd(tmp_fd, src, 0, 0, 3, 15);
        if (chk != 0)
        {
            abcdk_closep(&tmp_fd);
            remove(tmpfile);
            return NULL;
        }

        tmp_data = abcdk_object_mmap(tmp_fd, 0, 0, 0);

        // close and remove .
        abcdk_closep(&tmp_fd);
        remove(tmpfile);
    }
    else if (abcdk_strncmp(src, "file://", 7, 0) == 0)
    {
        tmp_data = abcdk_object_mmap_filename(src + 7, 0, 0, 0, 0);
    }
    else
    {
        tmp_data = abcdk_object_mmap_filename(src, 0, 0, 0, 0);
    }

    if (!tmp_data)
        return NULL;

    img = abcdk_xpu_imgcodec_decode(tmp_data->pptrs[0], tmp_data->sizes[0]);
    abcdk_object_unref(&tmp_data);

    return img;
}