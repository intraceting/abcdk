/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_IMAGE_HXX
#define ABCDK_XPU_NVIDIA_IMAGE_HXX

#include "abcdk/xpu/image.h"
#include "../base.in.h"
#include "../common/image.hxx"
#include "pixfmt.hxx"
#include "memory.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace image
        {
            typedef AVFrame _metadata;
            typedef _metadata metadata_t;

            void free(metadata_t **ctx);

            metadata_t *alloc();

            void clear(metadata_t *ctx);

            void zero(metadata_t *ctx);

            int get_buffer(metadata_t *ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align, int in_host);

            metadata_t *create(int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align, int in_host);

            int reset(metadata_t **ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align, int in_host);

            int copy(const metadata_t *src, int src_in_host, metadata_t *dst, int dst_in_host);

            int copy(const void *src_data, int src_linesize, int src_in_host, metadata_t *dst, int dst_plane, int dst_in_host);

            int copy(const metadata_t *src, int src_plane, int src_in_host, void *dst_data, int dst_linesize, int dst_in_host);

            int copy(const cv::Mat &src, int src_in_host, metadata_t *dst, int dst_in_host);

            metadata_t *clone(const metadata_t *src, int src_in_host, int dst_align, int dst_in_host);

            metadata_t *clone(abcdk_xpu_pixfmt_t pixfmt, const cv::Mat &src, int src_in_host, int dst_align, int dst_in_host);

            int get_width(const metadata_t *src);

            int get_height(const metadata_t *src);

            abcdk_xpu_pixfmt_t get_pixfmt(const metadata_t *src);

            int upload(const uint8_t *src_data[4], const int src_linesize[4], metadata_t *dst);

            int download(const metadata_t *src, uint8_t *dst_data[4], int dst_linesize[4]);
        } // namespace image
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_IMAGE_HXX