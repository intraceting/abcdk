/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_COMMON_CALIBRATE_HXX
#define ABCDK_XPU_COMMON_CALIBRATE_HXX

#include "abcdk/xpu/calibrate.h"
#include "../runtime.in.h"
#include "util.hxx"
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace common
    {
        class calibrate
        {
        public:
            /*图像尺寸(像素).*/
            cv::Size m_image_size;

            /*内参数矩阵.*/
            cv::Mat m_camera_matrix;

            /*5个畸变系数: k1,k2,p1,p2,k3. */
            cv::Mat m_dist_coeffs;

            /*x-remap.*/
            cv::Mat m_warper_xmap;

            /*y-remap.*/
            cv::Mat m_warper_ymap;
        public:
            static std::shared_ptr<calibrate> create();

        protected:
            calibrate();
            virtual ~calibrate();

        public:
            /**初始化.*/
            virtual void setup(const cv::Size &board_size, const cv::Size &grid_size) = 0;

            /**检测角点. */
            virtual int detect_corners(const cv::Mat &img, const cv::Size &win_size = cv::Size(15, 15)) = 0;

            /**
             * 评估参数.
             *
             * @return 重投影误差. 0~1, 数值越小表示相机参数越准确.
             */
            virtual double estimate_parameters() = 0;

            virtual int build_parameters(double alpha = 1) = 0;

            virtual int dump_parameters(std::string &data, const char *magic = NULL) = 0;
            virtual int load_parameters(const char *data,  const char *magic = NULL) = 0;
        };
    } // namespace common
} // namespace abcdk_xpu

#endif // ABCDK_XPU_COMMON_CALIBRATE_HXX