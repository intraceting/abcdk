/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_HXX
#define ABCDK_RTSP_SERVER_HXX

#include "abcdk/rtsp/live555.h"
#include "abcdk/util/rwlock.h"

#ifdef _RTSP_SERVER_HH

namespace abcdk
{
    namespace rtsp
    {
        class server : public RTSPServer
        {
        public:
            static server *createNew(UsageEnvironment &env, Port ourPort, UserAuthenticationDatabase *authDatabase, unsigned reclamationTestSeconds = 65)
            {
                int ourSocketIPv4 = setUpOurSocket(env, ourPort, AF_INET);
                int ourSocketIPv6 = setUpOurSocket(env, ourPort, AF_INET6);
                if (ourSocketIPv4 < 0 && ourSocketIPv6 < 0)
                    return NULL;

                return new server(env, ourSocketIPv4, ourSocketIPv6, ourPort, authDatabase, reclamationTestSeconds);
            }

        protected:
            server(UsageEnvironment &env, int ourSocketIPv4, int ourSocketIPv6, Port ourPort, UserAuthenticationDatabase *authDatabase, unsigned reclamationTestSeconds)
                : RTSPServer(env, ourSocketIPv4, ourSocketIPv6, ourPort, authDatabase, reclamationTestSeconds)
            {
            }

            virtual ~server()
            {
            }

        protected:
            virtual void lookupServerMediaSession(char const *streamName, lookupServerMediaSessionCompletionFunc *completionFunc, void *completionClientData, Boolean isFirstLookupInSession)
            {
            }
        };
    } // namespace rtsp
} // namespace abcdk

#endif //_RTSP_SERVER_HH

#endif // ABCDK_RTSP_SERVER_HXX