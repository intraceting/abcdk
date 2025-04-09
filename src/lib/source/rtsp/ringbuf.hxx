/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_RINGBUF_HXX
#define ABCDK_RTSP_RINGBUF_HXX

#include "abcdk/util/atomic.h"
#include "rwlock_robot.hxx"
#include "packet.hxx"

namespace abcdk
{
    namespace rtsp
    {
        class ringbuf
        {
        private:
            rtsp::rwlock m_queuq_locker;
            std::vector<packet> m_queue;
            uint64_t m_pos;

        public:
            ringbuf(int size = 100)
            {
                reset(size);
            }
            virtual ~ringbuf()
            {
            }

        public:
            void reset(int size)
            {
                assert(size > 0);

                rtsp::rwlock_robot autolock(&m_queuq_locker,1);

                m_queue.resize(size);
                m_pos = 0;
            }

            void write(packet &src)
            {
                rtsp::rwlock_robot autolock(&m_queuq_locker,0);

                m_queue[m_pos % m_queue.size()].copy_from(src);
                m_pos += 1;//next
            }

            void write(const void *data, size_t size, int64_t dts, int64_t pts, int64_t dur)
            {
                rtsp::rwlock_robot autolock(&m_queuq_locker,0);

                m_queue[m_pos % m_queue.size()].copy_from(data,size,dts,pts,dur);
                m_pos += 1;//next
            }

            int read(packet &dst, uint64_t &pos)
            {
                rtsp::rwlock_robot autolock(&m_queuq_locker,0);

                if (pos % m_queue.size() == m_pos % m_queue.size())
                    return 0;

                m_queue[pos % m_queue.size()].copy_to(dst);
                pos += 1;//next 

                return 1;
            }
        };
    } // namespace rtsp
} // namespace abcdk
#endif // ABCDK_RTSP_RINGBUF_HXX