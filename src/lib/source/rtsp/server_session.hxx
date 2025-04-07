/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_SESSION_HXX
#define ABCDK_RTSP_SERVER_SESSION_HXX

#include "abcdk/rtsp/rtsp.h"
#include "packet.hxx"

#ifdef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class session : public OnDemandServerMediaSubsession
        {
        protected:
            session(UsageEnvironment &env, Boolean reuseFirstSource, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
                : OnDemandServerMediaSubsession(env, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP)
            {
            }

            virtual ~session()
            {
                
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //_ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

#endif // ABCDK_RTSP_SERVER_SESSION_HXX
