/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_MEDIA_HXX
#define ABCDK_RTSP_SERVER_MEDIA_HXX

#include "abcdk/rtsp/live555.h"
#include "abcdk/util/rwlock.h"
#include "ringbuf.hxx"

#ifdef _SERVER_MEDIA_SESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class meida : public ServerMediaSession
        {
        private:
            std::map<int,rtsp::ringbuf*> m_buf; 
        public:
            static meida *createNew(UsageEnvironment &env, char const *streamName = NULL, char const *info = NULL, char const *description = NULL, Boolean isSSM = False, char const *miscSDPLines = NULL)
            {
                return new meida(env, streamName, info, description, isSSM, miscSDPLines);
            }

            static void deleteOld(meida **ctx)
            {
                meida *ctx_p;

                if(!ctx || !*ctx)
                    return ;
                
                ctx_p = *ctx;
                *ctx = NULL;

                ctx_p->deleteAllSubsessions();
                delete ctx_p;
            }

        public:
            meida(UsageEnvironment &env, char const *streamName, char const *info, char const *description, Boolean isSSM, char const *miscSDPLines)
                : ServerMediaSession(env, streamName, info, description, isSSM, miscSDPLines)
            {
            }

            virtual ~meida()
            {
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_SERVER_MEDIA_SESSION_HH

#endif // ABCDK_RTSP_SERVER_MEDIA_HXX