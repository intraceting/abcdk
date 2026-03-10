/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_IMAGE_HXX
#define ABCDK_XPU_GENERAL_IMAGE_HXX

#include "abcdk/xpu/image.h"
#include "../runtime.in.h"
#include "../common/image.hxx"
#include "pixfmt.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace image
        {
            typedef AVFrame _metadata;
            typedef _metadata metadata_t;

            void free(metadata_t **ctx);

            metadata_t *alloc();

            void clear(metadata_t *ctx);

            void zero(metadata_t *ctx);

            int get_buffer(metadata_t *ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align);

            metadata_t *create(int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align);

            int reset(metadata_t **ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align);
            
            int copy(const metadata_t *src, metadata_t *dst);

            int copy(const void *src_data, int src_linesize, metadata_t *dst, int dst_plane);

            int copy(const metadata_t *src, int src_plane, void *dst_data, int dst_linesize);

            int copy(const cv::Mat &src, metadata_t *dst);

            metadata_t *clone(const metadata_t *src, int dst_align);

            metadata_t *clone(abcdk_xpu_pixfmt_t pixfmt, const cv::Mat &src, int dst_align);

            int get_width(const metadata_t *src);

            int get_height(const metadata_t *src);

            abcdk_xpu_pixfmt_t get_pixfmt(const metadata_t *src);

            int upload(const uint8_t *src_data[4], const int src_linesize[4], metadata_t *dst);

            int download(const metadata_t *src, uint8_t *dst_data[4], int dst_linesize[4]);

        } // namespace image
    } // namespace general
} // namespace abcdk_xpu

#endif //ABCDK_XPU_GENERAL_IMAGE_HXX