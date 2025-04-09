/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_WAITER_HXX
#define ABCDK_RTSP_WAITER_HXX

#include "abcdk/rtsp/rtsp.h"
#include "abcdk/util/waiter.h"

namespace abcdk
{
    namespace rtsp
    {
        class waiter
        {
        private:
            abcdk_waiter_t *m_ctx;

        public:
            static void msg_destroy_for_string_cb(void *msg)
            {
                std::string *msg_p = (std::string *)msg;

                if (!msg_p)
                    return;

                delete msg_p;
            }

        public:
            waiter(abcdk_waiter_msg_destroy_cb cb = NULL)
            {
                m_ctx = abcdk_waiter_alloc(cb?cb:msg_destroy_for_string_cb);
            }
            virtual ~waiter()
            {
                abcdk_waiter_free(&m_ctx);
            }

        public:
            uint64_t reg()
            {
                uint64_t key = abcdk_sequence_num();
                int chk;

                chk = abcdk_waiter_register(m_ctx, key);
                if (chk != 0)
                    return 0;

                return key;
            }

            void *wait(uint64_t key, time_t timeout)
            {
                return abcdk_waiter_wait(m_ctx, key, timeout);
            }

            int response(uint64_t key, void *msg)
            {
                return abcdk_waiter_response(m_ctx, key, msg);
            }

            void cancel()
            {
                abcdk_waiter_cancel(m_ctx);
            }

            void resume()
            {
                abcdk_waiter_resume(m_ctx);
            }
        };
    } // namespace rtsp
} // namespace abcdk
#endif // ABCDK_RTSP_WAITER_HXX