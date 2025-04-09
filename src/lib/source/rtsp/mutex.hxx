/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_MUTEX_HXX
#define ABCDK_RTSP_MUTEX_HXX

#include "abcdk/rtsp/rtsp.h"
#include "abcdk/util/mutex.h"

namespace abcdk
{
    namespace rtsp
    {
        class mutex
        {
        private:
            abcdk_mutex_t *m_ctx;

        public:
            mutex()
            {
                m_ctx = abcdk_mutex_create();
            }
            virtual ~mutex()
            {
                abcdk_mutex_destroy(&m_ctx);
            }

        public:
            void lock(int block = 1)
            {
                abcdk_mutex_lock(m_ctx, block);
            }

            void unlock()
            {
                abcdk_mutex_unlock(m_ctx);
            }

            int wait(time_t timeout)
            {
                return abcdk_mutex_wait(m_ctx, timeout);
            }

            int signal(int broadcast)
            {
                return abcdk_mutex_signal(m_ctx, broadcast);
            }
        };
    } // namespace rtsp
} // namespace abcdk
#endif // ABCDK_RTSP_MUTEX_HXX