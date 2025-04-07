/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_RWLOCK_HXX
#define ABCDK_RTSP_RWLOCK_HXX

#include "abcdk/rtsp/live555.h"
#include "abcdk/util/rwlock.h"

namespace abcdk
{
    namespace rtsp
    {
        class rwlock
        {
        private:
            abcdk_rwlock_t *m_ctx;

        public:
            rwlock()
            {
                m_ctx = abcdk_rwlock_create();
            }
            virtual ~rwlock()
            {
                abcdk_rwlock_destroy(&m_ctx);
            }

        public:
            void rdlock(int block = 1)
            {
                abcdk_rwlock_rdlock(m_ctx, block);
            }

            void wrlock(int block = 1)
            {
                abcdk_rdlock_wrlock(m_ctx, block);
            }

            void lock(int block = 1)
            {
                wrlock(block);
            }

            void unlock()
            {
                abcdk_rwlock_unlock(m_ctx);
            }
        }
    } // namespace rtsp
} // namespace abcdk
#endif // ABCDK_RTSP_RWLOCK_HXX