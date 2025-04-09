/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_SOURCE_HXX
#define ABCDK_RTSP_SERVER_SOURCE_HXX

#include "abcdk/util/trace.h"
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
            int m_codec_id;

            TaskToken m_next_tasktoken;
            rtsp::packet m_pkt;

        public:
            int codec_id()
            {
                return m_codec_id;
            }
            
            void doGetNextFrame()
            {
                int chk;

                chk = fetch(m_pkt);
                if (chk <= 0)
                {
                    m_next_tasktoken = envir().taskScheduler().scheduleDelayedTask(10*1000, afterGetNextFrame, this);
                    return;
                }

                m_next_tasktoken = 0;

                //abcdk_trace_printf(LOG_DEBUG,"DTS(%lld),PTS(%lld),DUR(%lld),",m_pkt.dts(),m_pkt.pts(),m_pkt.dur());

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

                gettimeofday(&fPresentationTime, NULL);
                fDurationInMicroseconds = m_pkt.dur();

                memcpy(fTo, m_pkt.data(), fFrameSize);

                FramedSource::afterGetting(this);
            }
        protected:
            source(UsageEnvironment &env,int codec_id)
                : FramedSource(env)
            {
                m_codec_id = codec_id;
                m_next_tasktoken = 0;
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