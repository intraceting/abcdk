/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        namespace imgproc
        {
            cv::Mat find_homography(const abcdk_xpu_point_t src_quad[4], const abcdk_xpu_point_t dst_quad[4])
            {
                std::vector<cv::Point2f> tmp_dst_quad(4), tmp_src_quad(4);

                for (int i = 0; i < 4; i++)
                {
                    tmp_dst_quad[i].x = dst_quad[i].x;
                    tmp_dst_quad[i].y = dst_quad[i].y;
                }

                for (int i = 0; i < 4; i++)
                {
                    tmp_src_quad[i].x = src_quad[i].x;
                    tmp_src_quad[i].y = src_quad[i].y;
                }

                return cv::findHomography(tmp_src_quad, tmp_dst_quad, 0);
            }

            void find_homography(const abcdk_xpu_point_t src_quad[4], const abcdk_xpu_point_t dst_quad[4], abcdk_xpu_matrix_3x3_t *coeffs)
            {
                cv::Mat m = find_homography(src_quad, dst_quad);

                for (int y = 0; y < m.rows; y++)
                {
                    for (int x = 0; x < m.cols; x++)
                    {
                        coeffs->f64[y][x] = m.at<double>(y, x);
                    }
                }
            }

            cv::Mat _find_homography_face_112x112(float src[5][2])
            {
                float dst[5][2] = {{38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
                float avg0 = (src[0][0] + src[1][0] + src[2][0] + src[3][0] + src[4][0]) / 5;
                float avg1 = (src[0][1] + src[1][1] + src[2][1] + src[3][1] + src[4][1]) / 5;
                // Compute mean of src and dst.
                float src_mean[2] = {avg0, avg1};
                float dst_mean[2] = {56.0262f, 71.9008f};
                // Subtract mean from src and dst.
                float src_demean[5][2];
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 5; j++)
                    {
                        src_demean[j][i] = src[j][i] - src_mean[i];
                    }
                }
                float dst_demean[5][2];
                for (int i = 0; i < 2; i++)
                {
                    for (int j = 0; j < 5; j++)
                    {
                        dst_demean[j][i] = dst[j][i] - dst_mean[i];
                    }
                }
                double A00 = 0.0, A01 = 0.0, A10 = 0.0, A11 = 0.0;
                for (int i = 0; i < 5; i++)
                    A00 += dst_demean[i][0] * src_demean[i][0];
                A00 = A00 / 5;
                for (int i = 0; i < 5; i++)
                    A01 += dst_demean[i][0] * src_demean[i][1];
                A01 = A01 / 5;
                for (int i = 0; i < 5; i++)
                    A10 += dst_demean[i][1] * src_demean[i][0];
                A10 = A10 / 5;
                for (int i = 0; i < 5; i++)
                    A11 += dst_demean[i][1] * src_demean[i][1];
                A11 = A11 / 5;
                cv::Mat A = (cv::Mat_<double>(2, 2) << A00, A01, A10, A11);
                double d[2] = {1.0, 1.0};
                double detA = A00 * A11 - A01 * A10;
                if (detA < 0)
                    d[1] = -1;
                double T[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
                cv::Mat s, u, vt, v;
                cv::SVD::compute(A, s, u, vt);
                double smax = s.ptr<double>(0)[0] > s.ptr<double>(1)[0] ? s.ptr<double>(0)[0] : s.ptr<double>(1)[0];
                double tol = smax * 2 * FLT_MIN;
                int rank = 0;
                if (s.ptr<double>(0)[0] > tol)
                    rank += 1;
                if (s.ptr<double>(1)[0] > tol)
                    rank += 1;
                double arr_u[2][2] = {{u.ptr<double>(0)[0], u.ptr<double>(0)[1]}, {u.ptr<double>(1)[0], u.ptr<double>(1)[1]}};
                double arr_vt[2][2] = {{vt.ptr<double>(0)[0], vt.ptr<double>(0)[1]}, {vt.ptr<double>(1)[0], vt.ptr<double>(1)[1]}};
                double det_u = arr_u[0][0] * arr_u[1][1] - arr_u[0][1] * arr_u[1][0];
                double det_vt = arr_vt[0][0] * arr_vt[1][1] - arr_vt[0][1] * arr_vt[1][0];
                if (rank == 1)
                {
                    if ((det_u * det_vt) > 0)
                    {
                        cv::Mat uvt = u * vt;
                        T[0][0] = uvt.ptr<double>(0)[0];
                        T[0][1] = uvt.ptr<double>(0)[1];
                        T[1][0] = uvt.ptr<double>(1)[0];
                        T[1][1] = uvt.ptr<double>(1)[1];
                    }
                    else
                    {
                        double temp = d[1];
                        d[1] = -1;
                        cv::Mat D = (cv::Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
                        cv::Mat Dvt = D * vt;
                        cv::Mat uDvt = u * Dvt;
                        T[0][0] = uDvt.ptr<double>(0)[0];
                        T[0][1] = uDvt.ptr<double>(0)[1];
                        T[1][0] = uDvt.ptr<double>(1)[0];
                        T[1][1] = uDvt.ptr<double>(1)[1];
                        d[1] = temp;
                    }
                }
                else
                {
                    cv::Mat D = (cv::Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
                    cv::Mat Dvt = D * vt;
                    cv::Mat uDvt = u * Dvt;
                    T[0][0] = uDvt.ptr<double>(0)[0];
                    T[0][1] = uDvt.ptr<double>(0)[1];
                    T[1][0] = uDvt.ptr<double>(1)[0];
                    T[1][1] = uDvt.ptr<double>(1)[1];
                }
                double var1 = 0.0;
                for (int i = 0; i < 5; i++)
                    var1 += src_demean[i][0] * src_demean[i][0];
                var1 = var1 / 5;
                double var2 = 0.0;
                for (int i = 0; i < 5; i++)
                    var2 += src_demean[i][1] * src_demean[i][1];
                var2 = var2 / 5;
                double scale = 1.0 / (var1 + var2) * (s.ptr<double>(0)[0] * d[0] + s.ptr<double>(1)[0] * d[1]);
                double TS[2];
                TS[0] = T[0][0] * src_mean[0] + T[0][1] * src_mean[1];
                TS[1] = T[1][0] * src_mean[0] + T[1][1] * src_mean[1];
                T[0][2] = dst_mean[0] - scale * TS[0];
                T[1][2] = dst_mean[1] - scale * TS[1];
                T[0][0] *= scale;
                T[0][1] *= scale;
                T[1][0] *= scale;
                T[1][1] *= scale;
                cv::Mat transform_mat = (cv::Mat_<double>(2, 3) << T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
                return transform_mat;
            }

            cv::Mat find_homography_face_112x112(const abcdk_xpu_point_t face_kpt[5])
            {
                float src[5][2];

                for (int i = 0; i < 5; i++)
                {
                    src[i][0] = face_kpt[i].x;
                    src[i][1] = face_kpt[i].y;
                }

                return _find_homography_face_112x112(src);
            }

            void find_homography_face_112x112(const abcdk_xpu_point_t face_kpt[5], abcdk_xpu_matrix_3x3_t *coeffs)
            {
                cv::Mat m = find_homography_face_112x112(face_kpt);

                for (int y = 0; y < m.rows; y++)
                {
                    for (int x = 0; x < m.cols; x++)
                    {
                        coeffs->f64[y][x] = m.at<double>(y, x);
                    }
                }
            }

        } // namespace imgproc
    } // namespace common
} // namespace abcdk_xpu
