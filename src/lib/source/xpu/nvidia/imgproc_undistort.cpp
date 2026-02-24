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
            static int _undistort(const abcdk_xpu_size_t *size, double alpha,
                                  const abcdk_xpu_matrix_3x3_t *camera_matrix,
                                  const abcdk_xpu_scalar_t *dist_coeffs,
                                  image::metadata_t **xmap, image::metadata_t **ymap)
            {
                cv::Mat tmp_xmap, tmp_ymap;
                cv::Mat tmp_camera_matrix, tmp_dist_coeffs;
                int chk;

                tmp_camera_matrix = cv::Mat(3, 3, CV_64FC1, (void *)camera_matrix);
                tmp_dist_coeffs = cv::Mat(1, 5, CV_64FC1, (void *)dist_coeffs);

                chk = common::imgproc::undistort(size, alpha, tmp_camera_matrix, tmp_dist_coeffs, tmp_xmap, tmp_ymap);
                if (chk != 0)
                    return chk;

                *xmap = image::create(size->width, size->height, ABCDK_XPU_PIXFMT_GRAYF32, 16, 0);
                *ymap = image::create(size->width, size->height, ABCDK_XPU_PIXFMT_GRAYF32, 16, 0);

                if (!*xmap || !*ymap)
                {
                    image::free(xmap);
                    image::free(ymap);
                    return -ENOMEM;
                }

                image::copy(tmp_xmap.data, tmp_xmap.step, 1, *xmap, 0, 0);
                image::copy(tmp_ymap.data, tmp_ymap.step, 1, *ymap, 0, 0);

                return 0;
            }

            int undistort(const abcdk_xpu_size_t *size, double alpha,
                          const abcdk_xpu_matrix_3x3_t *camera_matrix,
                          const abcdk_xpu_scalar_t *dist_coeffs,
                          image::metadata_t **xmap, image::metadata_t **ymap)
            {

                return _undistort(size, alpha, camera_matrix, dist_coeffs, xmap, ymap);
            }

        } // namespace image
    } // namespace nvidia

} // namespace abcdk_xpu
