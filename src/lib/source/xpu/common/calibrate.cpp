/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "calibrate.hxx"

/*
 * 以下部分代码来自网络: https://blog.csdn.net/weixin_51229250/article/details/119976417
 */

namespace abcdk_xpu
{
    namespace common
    {
        class calibrate_impl : public calibrate
        {
        private:
            /*角点数量(行-1*列-1).*/
            cv::Size m_pattern_size;

            /*方格尺寸(毫米).*/
            cv::Size m_grid_size;

            /*原始角点列表.*/
            std::vector<std::vector<cv::Point2f>> m_pts_2d;

            /*三维角点列表.*/
            std::vector<std::vector<cv::Point3f>> m_pts_3d;

            /*每幅图像的旋转向量. */
            std::vector<cv::Mat> m_tvecs;

            /*每幅图像的平移向量. */
            std::vector<cv::Mat> m_rvecs;
        protected:
            bool m_camera_param_ok;
            bool m_undistort_param_ok;
        public:
            calibrate_impl()
            {
                m_camera_param_ok = false;
                m_undistort_param_ok = false;
            }

            virtual ~calibrate_impl()
            {
            }

        protected:
            void setup(const cv::Size &board_size = cv::Size(7, 11), const cv::Size &grid_size = cv::Size(25, 25))
            {
                assert(board_size.area() >= 4 && grid_size.area() >= 1);

                m_pattern_size = cv::Size(board_size.width - 1, board_size.height - 1);
                m_grid_size = grid_size;
                m_pts_2d.clear();
                m_pts_3d.clear();
                m_camera_matrix.release();
                m_dist_coeffs.release();
                m_tvecs.clear();
                m_rvecs.clear();

                m_image_size = cv::Size(0, 0);
                m_warper_xmap = cv::Mat();
                m_warper_ymap = cv::Mat();

                m_camera_param_ok = false;
                m_undistort_param_ok = false;
            }

            int detect_corners(const cv::Mat &img)
            {
                ABCDK_TRACE_ASSERT(!m_camera_param_ok, ABCDK_GETTEXT("评估参数已经建立, 不能检测角点."));

                ABCDK_TRACE_ASSERT(m_pattern_size.area() > 0, ABCDK_GETTEXT("标定板尺寸无效, 标定版未初始化."));
                ABCDK_TRACE_ASSERT(m_grid_size.area() > 0, ABCDK_GETTEXT("网络尺寸无效, 网络未初始化."));

                if (img.empty())
                    return -EINVAL;

                /*图像尺寸(像素), 所有图像尺寸必须统一.*/
                if (m_image_size.area() <= 0)
                {
                    m_image_size.width = img.cols;
                    m_image_size.height = img.rows;
                }

                if (img.cols != m_image_size.width || img.rows != m_image_size.height)
                {
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前图像尺寸与之前的图像尺寸不同, 忽略."));
                    return -EPERM;
                }

                /* 提取角点. */
                std::vector<cv::Point2f> pts_2d;
                if (!cv::findChessboardCorners(img, m_pattern_size, pts_2d))
                {
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("在当前图像中找不到角点或数量不足, 忽略."));
                    return -EPERM;
                }

                /*灰度化.*/
                cv::Mat img_gray;
                cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

                /* 亚像素精确化(对粗提取的角点进行精确化).*/
#if 0
			    cv::find4QuadCornerSubpix(img_gray, pts_2d, cv::Size(11, 11));
#else
                cv::Size sub_winsize(5, 5);
                cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
                cv::cornerSubPix(img_gray, pts_2d, sub_winsize, cv::Size(-1, -1), criteria);
#endif

                /*保存角点.*/
                m_pts_2d.push_back(pts_2d);

                const char *out_path_p = getenv("ABCDK_XPU_CALIBRATE_KEYPOINTS_DUMP_PATH");
                if (!out_path_p || !*out_path_p)
                    return 0;

                cv::Mat out = img.clone();

                for (auto &one : pts_2d)
                    cv::circle(out, cv::Point(one.x, one.y), 5, cv::Scalar(0, 0, 255));

                std::vector<char> out_file(PATH_MAX);
                sprintf(out_file.data(), "%s/c%zd.jpg", out_path_p,pts_2d.size());

                std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 100};
                cv::imwrite(out_file.data(), out, params);

                return 0;
            }

            double estimate_parameters()
            {
                ABCDK_TRACE_ASSERT(!m_camera_param_ok, ABCDK_GETTEXT("评估参数已经建立, 不能重复评估."));

                ABCDK_TRACE_ASSERT(m_pts_2d.size() >= 2, ABCDK_GETTEXT("有效图像不足, 至少需要两张有效图效."));

                /*清空.*/
                m_pts_3d.clear();

                for (int i = 0; i < m_pts_2d.size(); i++)
                {
                    /*务必保证图像中的有效角点数量与设定的数量一致.*/
                    assert(m_pts_2d[i].size() == m_pattern_size.area());

                    /*初始化标定板上角点的三维坐标. */

                    std::vector<cv::Point3f> pts_3d;
                    for (int y = 0; y < m_pattern_size.height; y++)
                    {
                        for (int x = 0; x < m_pattern_size.width; x++)
                        {
                            cv::Point3f pt;
                            /* 假设标定板放在世界坐标系中z=0的平面上.*/
                            pt.x = x * m_grid_size.width;
                            pt.y = y * m_grid_size.height;
                            pt.z = 0;
                            pts_3d.push_back(pt);
                        }
                    }

                    m_pts_3d.push_back(pts_3d);
                }

                /*
                 * 校准.
                 *
                 * RMS: 重投影误差. 三维世界的点投影到图像二维世界与实际检测到的图像点之间的平均偏差(像素单位). 数值越小, 表示相机参数越准确.
                 */
                double rms = cv::calibrateCamera(m_pts_3d, m_pts_2d, m_image_size, m_camera_matrix, m_dist_coeffs, m_rvecs, m_tvecs, 0);

                double total_offset = 0.0; /* 所有图像的平均误差的总和. */
                double offset = 0.0;       /* 每幅图像的平均误差. */

                for (int i = 0; i < m_pts_2d.size(); i++)
                {
                    /* 保存重新计算得到的投影点.*/
                    std::vector<cv::Point2f> pts_2d;

                    /* 通过得到的摄像机内外参数, 对空间的三维点进行重新投影计算, 得到新的投影点 */
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

                    abcdk_trace_printf(LOG_DEBUG, ABCDK_GETTEXT("第%d幅图像的平均误差(像素): %.6lf"), i + 1, offset);
                }

                abcdk_trace_printf(LOG_DEBUG, ABCDK_GETTEXT("总体平均误差(像素): %.6lf"), total_offset / m_pts_2d.size());

                m_camera_param_ok = true;//OK.

                return rms;
            }

            int build_parameters(double alpha = 1)
            {
                ABCDK_TRACE_ASSERT(!m_undistort_param_ok, ABCDK_GETTEXT("较正参数已经构建完成, 不能重复构建."));
                ABCDK_TRACE_ASSERT(m_camera_param_ok, ABCDK_GETTEXT("评估参数未建立或未加载."));

                abcdk_xpu_size_t img_size = {0};

                img_size.width = m_image_size.width;
                img_size.height = m_image_size.height;

                imgproc::undistort(&img_size,alpha,m_camera_matrix,m_dist_coeffs,m_warper_xmap,m_warper_ymap);

                m_undistort_param_ok = true; // OK.

                return 0;
            }

            int dump_parameters(std::string &dst, const char *magic = NULL)
            {
                cv::FileStorage f("{}", cv::FileStorage::MEMORY | cv::FileStorage::WRITE | cv::FileStorage::FORMAT_XML);
                if (!f.isOpened())
                    return -2;

                if (magic && *magic)
                    cv::write(f, "magic", magic);

                cv::write(f, "image_size", m_image_size);
                cv::write(f, "camera_matrix", m_camera_matrix);
                cv::write(f, "dist_coeffs", m_dist_coeffs);

                dst = f.releaseAndGetString();

                return 0;
            }

            int load_parameters(const char *src, const char *magic = NULL)
            {
                std::string old_magic;

                cv::FileStorage f(src, cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_XML);
                if (!f.isOpened())
                    return -1;
                
                cv::FileNode node = f["magic"];
                if (!node.empty())
                {
                    old_magic = node.string();
                    if (old_magic.compare(magic))
                        return -127;
                }
                else if (magic && *magic)
                {
                    return -127;
                }

                cv::FileNode size_node = f["image_size"];
                if (size_node.empty())
                    return -1;

                cv::FileNode camera_matrix_node = f["camera_matrix"];
                if (camera_matrix_node.empty())
                    return -1;

                cv::FileNode dist_coeffs_node = f["dist_coeffs"];
                if (dist_coeffs_node.empty())
                    return -1;

                m_image_size = cv::Size(size_node[0], size_node[1]);
                camera_matrix_node.mat().copyTo(m_camera_matrix);
                dist_coeffs_node.mat().copyTo(m_dist_coeffs);

                m_camera_param_ok = true;//OK.

                return 0;
            }

        };

        std::shared_ptr<calibrate> calibrate::create()
        {
            return std::make_shared<calibrate_impl>();
        }

        calibrate::calibrate()
        {
            ; // nothing to do;
        }

        calibrate::~calibrate()
        {
            ; // nothing to do;
        }
    } // namespace common
} // namespace abcdk_xpu
