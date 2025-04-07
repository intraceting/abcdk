/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_H265_SESSION_HXX
#define ABCDK_RTSP_SERVER_H265_SESSION_HXX

#include "abcdk/util/object.h"
#include "abcdk/util/basecode.h"
#include "abcdk/util/hevc.h"
#include "server_h265_source.hxx"
#include "server_session.hxx"

#ifdef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class h265_session : public session
        {
        private:
            rtsp::ringbuf *m_rgbuf_ctx_p;
            abcdk_hevc_extradata_t m_extdata;


            char *m_aux_sdp_line;
            char m_done_flag;
            RTPSink *m_dummy_rtp_sink;

        public:
            static h265_session *createNew(UsageEnvironment &env, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, Boolean reuseFirstSource = True, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
            {
                return new h265_session(env, rgbuf_ctx, extdata, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP);
            }

            void checkForAuxSDPLine1()
            {
                nextTask() = NULL;

                char const *dasl;
                if (m_aux_sdp_line != NULL)
                {
                    setDoneFlag();
                }
                else if (m_dummy_rtp_sink != NULL && (dasl = m_dummy_rtp_sink->auxSDPLine()) != NULL)
                {
                    m_aux_sdp_line = strDup(dasl);
                    m_dummy_rtp_sink = NULL;

                    setDoneFlag();
                }
                else if (!m_done_flag)
                {
                    int uSecsToDelay = 100000; // 100 ms
                    nextTask() = envir().taskScheduler().scheduleDelayedTask(uSecsToDelay, (TaskFunc *)checkForAuxSDPLine, this);
                }
            }

            void afterPlayingDummy1()
            {
                envir().taskScheduler().unscheduleDelayedTask(nextTask());
                setDoneFlag();
            }

        protected:
            h265_session(UsageEnvironment &env, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, Boolean reuseFirstSource = True, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
                : session(env, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP)
            {
                m_rgbuf_ctx_p = rgbuf_ctx;

                memset(&m_extdata,0,sizeof(m_extdata));
                abcdk_hevc_extradata_deserialize(extdata->pptrs[0], extdata->sizes[0],&m_extdata);

                m_aux_sdp_line = NULL;
                m_done_flag = 0;
                m_dummy_rtp_sink = NULL;
            }
            virtual ~h265_session()
            {
                for (int i = 0; i < m_extdata.nal_array_num; i++)
                    abcdk_object_unref(&m_extdata.nal_array[i].nal);

                delete[] m_aux_sdp_line;
            }

            static void afterPlayingDummy(void *clientData)
            {
                h265_session *subsess = (h265_session *)clientData;
                subsess->afterPlayingDummy1();
            }

            static void checkForAuxSDPLine(void *clientData)
            {
                h265_session *subsess = (h265_session *)clientData;
                subsess->checkForAuxSDPLine1();
            }

            void setDoneFlag()
            {
                m_done_flag = ~0;
            }

            virtual char const *getAuxSDPLine(RTPSink *rtpSink, FramedSource *inputSource)
            {
                if (m_aux_sdp_line != NULL)
                    return m_aux_sdp_line;

                if (m_dummy_rtp_sink == NULL)
                    m_dummy_rtp_sink = rtpSink;

                m_dummy_rtp_sink->startPlaying(*inputSource, afterPlayingDummy, this); // 获取sdp

                checkForAuxSDPLine(this);

                envir().taskScheduler().doEventLoop(&m_done_flag); // 程序在此等待

                return m_aux_sdp_line;
            }

            virtual FramedSource *createNewStreamSource(unsigned clientSessionId, unsigned &estBitrate)
            {
                h265_source *source_ctx = h265_source::createNew(envir(), m_rgbuf_ctx_p);
                if (!source_ctx)
                    return NULL;

                return H265VideoStreamDiscreteFramer::createNew(envir(), source_ctx);
            }

            virtual RTPSink *createNewRTPSink(Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource *inputSource)
            {
                abcdk_object_t *vps_p = NULL, *sps_p = NULL, *pps_p = NULL;

                for (int i = 0; i < m_extdata.nal_array_num; i++)
                {
                    if (m_extdata.nal_array[i].unit_type == 32)
                        vps_p = m_extdata.nal_array[i].nal;
                    else if (m_extdata.nal_array[i].unit_type == 33)
                        sps_p = m_extdata.nal_array[i].nal;
                    else if (m_extdata.nal_array[i].unit_type == 34)
                        pps_p = m_extdata.nal_array[i].nal;
                }

                if(!vps_p || !sps_p || !pps_p)
                    return NULL;

                return H265VideoRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic, vps_p->pptrs[0], vps_p->sizes[0], sps_p->pptrs[0], sps_p->sizes[0], pps_p->pptrs[0], pps_p->sizes[0]);
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //_ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

#endif // ABCDK_RTSP_SERVER_H265_SESSION_HXX
