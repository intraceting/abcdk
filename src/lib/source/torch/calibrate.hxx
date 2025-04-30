/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_CALIBRATE_HXX
#define ABCDK_TORCH_CALIBRATE_HXX

#include "abcdk/util/trace.h"
#include "abcdk/torch/opencv.h"

#ifdef OPENCV_CALIB3D_HPP

/*
 * 以下部分代码来自网络：https://blog.csdn.net/weixin_51229250/article/details/119976417
*/


namespace abcdk
{
    namespace torch
    {
        class calibrate
        {
        private:
            /*角点数量(行-1*列-1)。*/
            cv::Size m_pattern_size;

            /*方格尺寸(毫米)。*/
            cv::Size m_grid_size;

            /*图像尺寸(像素)。*/
            cv::Size m_image_size;

            /*原始角点列表。*/
            std::vector<std::vector<cv::Point2f>> m_pts_2d;

            /*三维角点列表。*/
            std::vector<std::vector<cv::Point3f>> m_pts_3d;

            /*内参数矩阵。*/
            cv::Mat m_camera_matrix;

            /*5个畸变系数：k1,k2,p1,p2,k3。 */
            cv::Mat m_dist_coeffs;

            /*每幅图像的旋转向量。 */
            std::vector<cv::Mat> m_tvecs;

            /*每幅图像的平移向量。 */
            std::vector<cv::Mat> m_rvecs;

        public:
            calibrate()
            {
                Setup();
            }

            virtual ~calibrate()
            {
            }

        public:
            void Setup(const cv::Size &board_size = cv::Size(7, 10), const cv::Size &grid_size = cv::Size(25, 25))
            {
                assert(board_size.area() >= 4 && grid_size.area() >= 1);

                m_pattern_size = cv::Size(board_size.width - 1, board_size.height - 1);
                m_grid_size = grid_size;
                m_image_size = cv::Size(0, 0);
                m_pts_2d.clear();
                m_pts_3d.clear();
                m_camera_matrix.release();
                m_dist_coeffs.release();
                m_tvecs.clear();
                m_rvecs.clear();
            }

            size_t Bind(const cv::Mat &img)
            {
                return FindCorners(img);
            }

            double Estimate()
            {
                double chk_rms;

                chk_rms = EstimateCamera();

                AppraiseCamera();

                return chk_rms;
            }

            void GetParam(cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
            {
                assert(m_camera_matrix.depth() == CV_64F);
                assert(m_dist_coeffs.depth() == CV_64F);

                /*复制参数。*/
                camera_matrix = m_camera_matrix;
                dist_coeffs = m_dist_coeffs;

            }

            void GetRectifyMap(double alpha, cv::Mat &xmap, cv::Mat &ymap)
            {
                assert(m_camera_matrix.depth() == CV_64F);
                assert(m_dist_coeffs.depth() == CV_64F);

                cv::Mat R = cv::Mat::eye(3, 3, CV_64F); // 不做旋转
                cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(m_camera_matrix, m_dist_coeffs, m_image_size, alpha, m_image_size, 0);

                // 生成映射表，注意 xmap 和 ymap 都是 CV_32FC1（CUDA兼容）
                cv::initUndistortRectifyMap(m_camera_matrix, m_dist_coeffs, R, newCameraMatrix, m_image_size, CV_32FC1, xmap, ymap);
            }

        protected:

            size_t FindCorners(const cv::Mat &img)
            {
                assert(m_pattern_size.area() > 0);
                assert(m_grid_size.area() > 0);

                if (img.empty())
                    return m_pts_2d.size();

                /*图像尺寸(像素)。所有图像尺寸必须统一。*/
                if (m_image_size.area() <= 0)
                {
                    m_image_size.width = img.cols;
                    m_image_size.height = img.rows;
                }

                if (img.cols != m_image_size.width || img.rows != m_image_size.height)
                {
                    abcdk_trace_printf(LOG_WARNING, TT("当前图像尺寸与之前的图像尺寸不同，忽略。"));
                    return m_pts_2d.size();
                }

                /* 提取角点。 */
                std::vector<cv::Point2f> pts_2d;
                if (!cv::findChessboardCorners(img, m_pattern_size, pts_2d))
                {
                    abcdk_trace_printf(LOG_WARNING, TT("在当前图像中找不到角点或数量不足，忽略。"));
                    return m_pts_2d.size();
                }

                /*灰度化。*/
                cv::Mat img_gray;
                cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

                /* 亚像素精确化(对粗提取的角点进行精确化)。*/
#if 0
			    cv::find4QuadCornerSubpix(img_gray, pts_2d, cv::Size(11, 11));
#else
                cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
                cv::cornerSubPix(img_gray, pts_2d, cv::Size(5, 5), cv::Size(-1, -1), criteria);
#endif

                /*保存角点。*/
                m_pts_2d.push_back(pts_2d);

                return m_pts_2d.size();
            }

            double EstimateCamera()
            {
                assert(m_pattern_size.area() > 0);
                assert(m_grid_size.area() > 0);
                assert(m_pts_2d.size() >= 2);

                /*清空。*/
                m_pts_3d.clear();

                for (int i = 0; i < m_pts_2d.size(); i++)
                {
                    /*务必保证图像中的有效角点数量与设定的数量一致。*/
                    assert(m_pts_2d[i].size() == m_pattern_size.area());

                    /*初始化标定板上角点的三维坐标。 */

                    std::vector<cv::Point3f> pts_3d;
                    for (int y = 0; y < m_pattern_size.height; y++)
                    {
                        for (int x = 0; x < m_pattern_size.width; x++)
                        {
                            cv::Point3f pt;
                            /* 假设标定板放在世界坐标系中z=0的平面上。*/
                            pt.x = x * m_grid_size.width;
                            pt.y = y * m_grid_size.height;
                            pt.z = 0;
                            pts_3d.push_back(pt);
                        }
                    }

                    m_pts_3d.push_back(pts_3d);
                }



                /* 
                 * 校准。
                 *
                 * RMS：重投影误差。校准后的相机参数，把三维世界点投影到图像平面后，与实际检测到的图像点之间的平均偏差（像素单位）。数值越小，表示相机模型越准确，标定质量越好。
                 */
                double RMS = cv::calibrateCamera(m_pts_3d, m_pts_2d, m_image_size, m_camera_matrix, m_dist_coeffs, m_rvecs, m_tvecs, 0);

                return RMS;
            }

            void AppraiseCamera()
            {
                assert(m_pattern_size.area() > 0);
                assert(m_grid_size.area() > 0);
                assert(m_pts_2d.size() >= 2 && m_pts_3d.size() >= 2);
                assert(m_rvecs.size() >= 2 && m_tvecs.size() >= 2);
                assert(m_pts_2d.size() == m_pts_3d.size());
                assert(m_pts_2d.size() == m_rvecs.size());
                assert(m_pts_2d.size() == m_tvecs.size());
                assert(!m_camera_matrix.empty() && !m_dist_coeffs.empty());

                double total_offset = 0.0;         /* 所有图像的平均误差的总和。 */
                double offset = 0.0;               /* 每幅图像的平均误差。 */

                for (int i = 0; i < m_pts_2d.size(); i++)
                {
                    /* 保存重新计算得到的投影点。*/
                    std::vector<cv::Point2f> pts_2d; 

                    /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
                    cv::projectPoints(m_pts_3d[i], m_rvecs[i], m_tvecs[i], m_camera_matrix, m_dist_coeffs, pts_2d);

                    /* 计算新的投影点和旧的投影点之间的误差*/
                    std::vector<cv::Point2f> tempImagePoint = m_pts_2d[i];
                    cv::Mat tempImagePointMat = cv::Mat(1, tempImagePoint.size(), CV_32FC2);
                    cv::Mat image_points2Mat = cv::Mat(1, pts_2d.size(), CV_32FC2);

                    for (int j = 0; j < tempImagePoint.size(); j++)
                    {
                        image_points2Mat.at<cv::Vec2f>(0, j) = cv::Vec2f(pts_2d[j].x, pts_2d[j].y);
                        tempImagePointMat.at<cv::Vec2f>(0, j) = cv::Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
                    }
                    offset = cv::norm(image_points2Mat, tempImagePointMat, cv::NORM_L2);
                    total_offset += (offset /= m_pts_2d[i].size());
            
                    abcdk_trace_printf(LOG_DEBUG, TT("第%d幅图像的平均误差(像素)：%.6lf"), i + 1, offset);
                }
            
                abcdk_trace_printf(LOG_DEBUG, TT("总体平均误差(像素)：%.6lf "), total_offset / m_pts_2d.size());
            }
        };
    } // namespace torch
} // namespace abcdk

#endif // OPENCV_CALIB3D_HPP

#endif // ABCDK_TORCH_CALIBRATE_HXX