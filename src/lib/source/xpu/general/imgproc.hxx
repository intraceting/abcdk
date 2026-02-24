/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_IMGPROC_HXX
#define ABCDK_XPU_GENERAL_IMGPROC_HXX

#include "abcdk/xpu/imgproc.h"
#include "../runtime.in.h"
#include "../common/imgproc.hxx"
#include "../common/util.hxx"
#include "image.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace imgproc
        {
            static inline cv::InterpolationFlags inter_local_to_opencv(abcdk_xpu_inter_t mode)
            {
                return common::imgproc::inter_local_to_opencv(mode);
            }

            static inline cv::InterpolationFlags inter_local_to_opencv(int mode)
            {
                return common::imgproc::inter_local_to_opencv((abcdk_xpu_inter_t)mode);
            }

            static inline int convert(const image::metadata_t *src, image::metadata_t *dst)
            {
                return common::imgproc::convert(src,dst);
            }

            int resize(const image::metadata_t *src, const abcdk_xpu_rect_t *src_roi, image::metadata_t *dst, abcdk_xpu_inter_t inter_mode);

            int stuff(image::metadata_t *dst, const abcdk_xpu_rect_t *roi, const abcdk_xpu_scalar_t *scalar);

            int warp(const image::metadata_t *src, image::metadata_t *dst, const abcdk_xpu_matrix_3x3_t *coeffs, int warp_mode, abcdk_xpu_inter_t inter_mode);

            int warp_quad2quad(const image::metadata_t *src, const abcdk_xpu_point_t src_quad[4],
                               image::metadata_t *dst, const abcdk_xpu_point_t dst_quad[4],
                               int warp_mode, abcdk_xpu_inter_t inter_mode);

            int undistort(const abcdk_xpu_size_t *size, double alpha,
                          const abcdk_xpu_matrix_3x3_t *camera_matrix,
                          const abcdk_xpu_scalar_t *dist_coeffs,
                          image::metadata_t **xmap, image::metadata_t **ymap);

            int remap(const image::metadata_t *src, image::metadata_t *dst, 
                      const image::metadata_t *xmap, const image::metadata_t *ymap,
                      abcdk_xpu_inter_t inter_mode);

            int line(image::metadata_t *dst, const abcdk_xpu_point_t *p1, const abcdk_xpu_point_t *p2,
                     const abcdk_xpu_scalar_t *color, int weight);

            int rectangle(image::metadata_t *dst, const abcdk_xpu_rect_t *rect, int weight, const abcdk_xpu_scalar_t *color);

            int brightness(image::metadata_t *dst, const abcdk_xpu_scalar_t *alpha, const abcdk_xpu_scalar_t *bate);

            int compose(image::metadata_t *panorama, const image::metadata_t *part,
                        size_t overlap_x, size_t overlap_y, size_t overlap_w,
                        const abcdk_xpu_scalar_t *scalar, int optimize_seam);

            int mask(image::metadata_t *dst, const image::metadata_t *src, float threshold, const abcdk_xpu_scalar_t *color, int less_or_not);

            int blob_8u_to_32f(int dst_packed, float *dst, size_t dst_ws, int dst_c_invert,
                               int src_packed, const uint8_t *src, size_t src_ws, int src_c_invert,
                               size_t b, size_t w, size_t h, size_t c,
                               const abcdk_xpu_scalar_t *scale, const abcdk_xpu_scalar_t *mean, const abcdk_xpu_scalar_t *std);

            int blob_32f_to_8u(int dst_packed, uint8_t *dst, size_t dst_ws, int dst_c_invert,
                               int src_packed, const float *src, size_t src_ws, int src_c_invert,
                               size_t b, size_t w, size_t h, size_t c,
                               const abcdk_xpu_scalar_t *scale, const abcdk_xpu_scalar_t *mean, const abcdk_xpu_scalar_t *std);
                               
            int quad2rect(const image::metadata_t *src, const abcdk_xpu_point_t src_quad[4], image::metadata_t *dst, abcdk_xpu_inter_t inter_mode);

            int face_warp(const image::metadata_t *src, const abcdk_xpu_point_t face_kpt[5], image::metadata_t **dst, abcdk_xpu_inter_t inter_mode);
        } // namespace image
    } // namespace general
} // namespace abcdk_xpu

#endif //ABCDK_XPU_GENERAL_IMGPROC_HXX