/**
 * 源码来自网络，略有改动。 
*/

#pragma once

#include "STrack.h"

#if defined(OPENCV_CORE_HPP) && defined(EIGEN_CORE_H)

#ifndef __BYTETRACK__
#define __BYTETRACK__
#endif //__BYTETRACK__

namespace bytetrack
{
    struct object
    {
        int x;
        int y;
        int w;
        int h;
        int label;
        float score;
        uint64_t magic;
    };

    class BYTETracker
    {
    public:
        BYTETracker(int _max_time_lost, float _track_thresh = 0.5, float _high_thresh = 0.6, float _match_thresh = 0.8);
        BYTETracker(int frame_rate, int track_buffer); // 30,30
        virtual ~BYTETracker();

    public:
        vector<STrack> update(const vector<object> &objects);
        Scalar get_color(int idx);

    private:
        vector<STrack *> joint_stracks(vector<STrack *> &tlista, vector<STrack> &tlistb);
        vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

        vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
        void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

        void linear_assignment(vector<vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
                               vector<vector<int>> &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
        vector<vector<float>> iou_distance(vector<STrack *> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
        vector<vector<float>> iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
        vector<vector<float>> ious(vector<vector<float>> &atlbrs, vector<vector<float>> &btlbrs);

        double lapjv(const vector<vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol,
                     bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

    private:
        float track_thresh;
        float high_thresh;
        float match_thresh;
        int frame_id;
        int max_time_lost;

        vector<STrack> tracked_stracks;
        vector<STrack> lost_stracks;
        vector<STrack> removed_stracks;
        byte_kalman::KalmanFilter kalman_filter;
    };

} // bytetrack

#endif //#if defined(OPENCV_CORE_HPP) && defined(EIGEN_CORE_H)