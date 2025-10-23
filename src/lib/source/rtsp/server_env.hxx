/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_ENV_HXX
#define ABCDK_RTSP_SERVER_ENV_HXX

#include "abcdk/rtsp/rtsp.h"

#ifdef HAVE_LIVE555

namespace abcdk
{
    namespace rtsp_server
    {
        class env : public BasicUsageEnvironment
        {
        public:
            static env *createNew(TaskScheduler &taskScheduler)
            {
                return new env(taskScheduler);
            }

            static void deleteOld(env **ctx)
            {
                env *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                delete ctx_p;
            }

        protected:
            env(TaskScheduler &taskScheduler)
                : BasicUsageEnvironment(taskScheduler)
            {
            }

            virtual ~env()
            {
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //HAVE_LIVE555

#endif // ABCDK_RTSP_SERVER_ENV_HXX