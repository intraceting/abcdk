/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_RWLOCK_ROBOT_HXX
#define ABCDK_RTSP_RWLOCK_ROBOT_HXX

#include "rwlock.hxx"

namespace abcdk
{
    namespace rtsp
    {
        /*读写锁机器人, 自动执行入栈和出栈.*/
        class rwlock_robot
        {
        private:
            rwlock *m_ctx_p;
        public:
            rwlock_robot(rwlock *ctx, int writer)
            {
                m_ctx_p = ctx;

                if (writer)
                    m_ctx_p->wrlock();
                else
                    m_ctx_p->rdlock();
            }

            virtual ~rwlock_robot()
            {
                m_ctx_p->unlock();
            }
        };

    } // namespace torch_cuda
} // namespace abcdk


#endif // ABCDK_RTSP_RWLOCK_ROBOT_HXX