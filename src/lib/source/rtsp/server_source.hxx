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

#ifdef HAVE_LIVE555

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
            size_t m_pkt_offset;

        public:
            int codec_id()
            {
                return m_codec_id;
            }
            
            void doGetNextFrame()
            {
                int chk;

                if (m_pkt.size() == 0 || m_pkt.size() == m_pkt_offset)
                {
                    chk = fetch(m_pkt);
                    if (chk <= 0)
                    {
                        m_next_tasktoken = envir().taskScheduler().scheduleDelayedTask(10 * 1000, afterGetNextFrame, this); // 100fps
                        return;
                    }

                    m_next_tasktoken = 0;
                    m_pkt_offset = 0;//must to 0.
                }

                if (m_pkt.size(m_pkt_offset) > fMaxSize)
                {
                    fFrameSize = fMaxSize;
                    fNumTruncatedBytes = m_pkt.size(m_pkt_offset) - fMaxSize;
                }
                else
                {
                    fFrameSize = m_pkt.size(m_pkt_offset);
                    fNumTruncatedBytes = 0;
                }

                memcpy(fTo, m_pkt.data(m_pkt_offset), fFrameSize);
                m_pkt_offset += fFrameSize;

#if 1
                gettimeofday(&fPresentationTime, NULL);
#else 
                fPresentationTime.tv_sec = m_pkt.pts()/1000000;
                fPresentationTime.tv_usec = m_pkt.pts()%1000000;
#endif
                
                fDurationInMicroseconds = m_pkt.dur();


                FramedSource::afterGetting(this);
            }
        protected:
            source(UsageEnvironment &env,int codec_id)
                : FramedSource(env)
            {
                m_codec_id = codec_id;
                m_next_tasktoken = 0;

                m_pkt_offset = 0;
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

#endif //HAVE_LIVE555

#endif // ABCDK_RTSP_SERVER_SOURCE_HXX