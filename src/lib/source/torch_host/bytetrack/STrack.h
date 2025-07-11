/**
 * 源码来自网络，略有改动。 
*/


#pragma once

#include "kalmanFilter.h"

#if defined(OPENCV_CORE_HPP) && defined(EIGEN_CORE_H)

using namespace cv;
using namespace std;

namespace bytetrack
{

    enum TrackState
    {
        New = 0,
        Tracked,
        Lost,
        Removed
    };

    class STrack
    {
    public:
        STrack(vector<float> tlwh_, float score, uint64_t magic);
        ~STrack();

        vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
        void static multi_predict(vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter);
        void static_tlwh();
        void static_tlbr();
        vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
        vector<float> to_xyah();
        void mark_lost();
        void mark_removed();
        int next_id();
        int end_frame();

        void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
        void re_activate(STrack &new_track, int frame_id, bool new_id = false);
        void update(STrack &new_track, int frame_id);

    public:
        bool is_activated;
        int track_id;
        int state;

        vector<float> _tlwh;
        vector<float> tlwh;
        vector<float> tlbr;
        int frame_id;
        int tracklet_len;
        int start_frame;

        KAL_MEAN mean;
        KAL_COVA covariance;
        float score;

        // 魔法数，用于输入和输出的匹配。
        uint64_t m_magic;

    private:
        byte_kalman::KalmanFilter kalman_filter;
    };

} // aicontirb

#endif //#if defined(OPENCV_CORE_HPP) && defined(EIGEN_CORE_H)