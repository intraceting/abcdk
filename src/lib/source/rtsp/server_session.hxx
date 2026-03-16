/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_SESSION_HXX
#define ABCDK_RTSP_SERVER_SESSION_HXX

#include "abcdk/rtsp/rtsp.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/socket.h"
#include "packet.hxx"

#ifdef HAVE_LIVE555

namespace abcdk
{
    namespace rtsp_server
    {
        class session : public OnDemandServerMediaSubsession
        {
        private:
            int m_codec_id;

        public:
            int codec_id()
            {
                return m_codec_id;
            }

        protected:
            session(UsageEnvironment &env, int codec_id, Boolean reuseFirstSource = True, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
                : OnDemandServerMediaSubsession(env, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP)
            {
                m_codec_id = codec_id;
            }

            virtual ~session()
            {
                
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //HAVE_LIVE555

#endif // ABCDK_RTSP_SERVER_SESSION_HXX
