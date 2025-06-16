/**
 * 源码来自网络，略有改动。 
*/

#pragma once

#include <cstddef>
#include <vector>
#include <fstream>

#ifdef HAVE_OPENCV
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#endif //__cplusplus
#endif //HAVE_OPENCV

#ifdef HAVE_EIGEN
#ifdef __cplusplus
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#endif //__cplusplus
#endif //HAVE_EIGEN

#if defined(OPENCV_CORE_HPP) && defined(EIGEN_CORE_H)

namespace bytetrack
{

    typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
    typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
    typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
    typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;
    // typedef std::vector<FEATURE> FEATURESS;

    // Kalmanfilter
    // typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
    typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
    typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
    typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
    typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
    using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
    using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

    // main
    using RESULT_DATA = std::pair<int, DETECTBOX>;

    // tracker:
    using TRACKER_DATA = std::pair<int, FEATURESS>;
    using MATCH_DATA = std::pair<int, int>;
    typedef struct t
    {
        std::vector<MATCH_DATA> matches;
        std::vector<int> unmatched_tracks;
        std::vector<int> unmatched_detections;
    } TRACHER_MATCHD;

    // linear_assignment:
    typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;

} // bytetrack

#endif //#if defined(OPENCV_CORE_HPP) && defined(EIGEN_CORE_H)