/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_SOURCE_HXX
#define ABCDK_RTSP_SERVER_SOURCE_HXX

#include "abcdk/rtsp/rtsp.h"
#include "packet.hxx"

#ifdef _FRAMED_SOURCE_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class source : public FramedSource
        {
        private:
            TaskToken m_next_tasktoken;
            rtsp::packet m_pkt;

        public:
            void doGetNextFrame()
            {
                int chk;

                chk = fetch(m_pkt);
                if (chk <= 0)
                {
                    m_next_tasktoken = envir().taskScheduler().scheduleDelayedTask(10 * 1000, afterGetNextFrame, this);
                    return;
                }

                m_next_tasktoken = 0;

                if (m_pkt.size() > fMaxSize)
                {
                    fFrameSize = fMaxSize;
                    fNumTruncatedBytes = m_pkt.size() - fMaxSize;
                }
                else
                {
                    fFrameSize = m_pkt.size();
                    fNumTruncatedBytes = 0;
                }

                fPresentationTime = *m_pkt.time();
                fDurationInMicroseconds = m_pkt.dur();

                memcpy(fTo, m_pkt.data(), fFrameSize);

                FramedSource::afterGetting(this);
            }
        protected:
            source(UsageEnvironment &env)
                : FramedSource(env)
            {
            }
            virtual ~source()
            {
                envir().taskScheduler().unscheduleDelayedTask(m_next_tasktoken);
            }

            static void afterGetNextFrame(void *data)
            {
                source *ctx_p = (source *)data;
                ctx_p->doGetNextFrame();
            }

            virtual int fetch(rtsp::packet &pkt) = 0;
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_FRAMED_SOURCE_HH

#endif // ABCDK_RTSP_SERVER_SOURCE_HXX