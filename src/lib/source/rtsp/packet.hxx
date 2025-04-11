/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_PACKET_HXX
#define ABCDK_RTSP_PACKET_HXX

#include "abcdk/rtsp/rtsp.h"
#include "abcdk/util/object.h"

namespace abcdk
{
    namespace rtsp
    {
        class packet
        {
        private:
            uint8_t *m_data;
            size_t m_size;

            int64_t m_dur; // 时长(微秒)。

            abcdk_object_t *m_buf;
            
        public:
            packet(const void *data = NULL, size_t size = 0, int64_t dur = 0)
            {
                clear(true);

                m_data = (uint8_t*)data;
                m_size = size;
                m_dur = dur;
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
            uint8_t *data() const
            {
                return m_data;
            }

            size_t size() const
            {
                return m_size;
            }

            int64_t dur() const
            {
                return m_dur;
            }

        public:
            void clear(bool first = false)
            {
                m_data = NULL;
                m_size = 0;
                m_dur = 0;

                if (first)
                    m_buf = NULL;
                else
                    abcdk_object_unref(&m_buf);
            }

            packet &operator=(const packet &src)
            {
                clear();

                m_buf = abcdk_object_copyfrom(src.data(),src.size());
                m_dur = src.dur();

                m_data = m_buf->pptrs[0];
                m_size = m_buf->sizes[0];
                
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

            void copy_from(const void *data, size_t size, int64_t dur)
            {
                packet src(data,size,dur);

                *this = src;
            }
        };
    } // namespace rtsp
} // namespace abcdk
#endif // ABCDK_RTSP_PACKET_HXX