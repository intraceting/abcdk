/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_PACKET_HXX
#define ABCDK_RTSP_PACKET_HXX

#include "abcdk/rtsp/rtsp.h"

namespace abcdk
{
    namespace rtsp
    {
        class packet
        {
        private:
            std::vector<uint8_t> m_buf;
            int64_t m_pts;
            int64_t m_dts;
            int64_t m_dur; // 时长(微秒)。
        public:
            packet()
            {
                clear();
            }
            packet(const packet &src)
            {
                copy_from(src);
            }
            virtual ~packet()
            {
                clear();
            }

        public:
            uint8_t *data()
            {
                return m_buf.data();
            }

            size_t size()
            {
                return m_buf.size();
            }

            int64_t pts()
            {
                return m_pts;
            }

            int64_t dts()
            {
                return m_dts;
            }

            int64_t dur()
            {
                return m_dur;
            }

        public:
            void clear()
            {
                m_buf.clear();
                m_pts = (int64_t)UINT64_C(0x8000000000000000);
                m_dts = (int64_t)UINT64_C(0x8000000000000000);
                m_dur = 0;
            }

            packet &operator=(const packet &src)
            {
                m_buf = src.m_buf;
                m_dts = src.m_dts;
                m_pts = src.m_pts;
                m_dur = src.m_dur;

                return *this;
            }

            void copy_from(const packet &src)
            {
                *this = src;
            }

            void copy_to(packet &dst)
            {
                dst = *this;
            }

            void copy_from(const void *data, size_t size, int64_t dts, int64_t pts, int64_t dur)
            {
                m_buf.resize(size);
                memcpy(m_buf.data(),data,size);

                m_dts = dts;
                m_pts = pts;
                m_dur = dur;
            }
        };
    } // namespace rtsp
} // namespace abcdk
#endif // ABCDK_RTSP_PACKET_HXX