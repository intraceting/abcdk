/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace nvidia
    {
        namespace imgproc
        {
            static int _face_warp(const image::metadata_t *src, const abcdk_xpu_point_t face_kpt[5], image::metadata_t **dst, abcdk_xpu_inter_t inter_mode)
            {
                abcdk_xpu_matrix_3x3_t coeffs;
                int chk;

                common::imgproc::find_homography_face_112x112(face_kpt, &coeffs);

                chk = image::reset(dst, 112, 112, pixfmt::ffmpeg_to_local(src->format), 16, 0);
                if (chk != 0)
                    return -1;

                return warp(src, *dst, &coeffs, 2, inter_mode);
            }

            int face_warp(const image::metadata_t *src, const abcdk_xpu_point_t face_kpt[5], image::metadata_t **dst, abcdk_xpu_inter_t inter_mode)
            {
                assert(src->format == AV_PIX_FMT_GRAY8 ||
                       src->format == AV_PIX_FMT_RGB24 ||
                       src->format == AV_PIX_FMT_BGR24 ||
                       src->format == AV_PIX_FMT_RGB32 ||
                       src->format == AV_PIX_FMT_BGR32);

                return _face_warp(src, face_kpt, dst, inter_mode);
            }

        } // namespace image
    } // namespace nvidia
} // namespace abcdk_xpu
