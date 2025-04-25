/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_H264_SESSION_HXX
#define ABCDK_RTSP_SERVER_H264_SESSION_HXX

#include "abcdk/util/object.h"
#include "abcdk/util/basecode.h"
#include "abcdk/util/h264.h"
#include "server_h264_source.hxx"
#include "server_session.hxx"

#ifdef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class h264_session : public session
        {
        private:
            rtsp::ringbuf *m_rgbuf_ctx_p;
            abcdk_h264_extradata_t m_extdata;
            uint32_t m_bitrate;

            char *m_aux_sdp_line;
            char m_done_flag;
            RTPSink *m_dummy_rtp_sink;

        public:
            static h264_session *createNew(UsageEnvironment &env,int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, uint32_t bitrate)
            {
                return new h264_session(env, codec_id, rgbuf_ctx, extdata, bitrate);
            }

            static void deleteOld(h264_session **ctx)
            {
                h264_session *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                Medium::close(ctx_p->envir(),ctx_p->name());
                //delete ctx_p;
            }

        protected:
            h264_session(UsageEnvironment &env,int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, uint32_t bitrate)
                : session(env, codec_id)
            {
                m_rgbuf_ctx_p = rgbuf_ctx;
                
                memset(&m_extdata,0,sizeof(m_extdata));
                abcdk_h264_extradata_deserialize(extdata->pptrs[0], extdata->sizes[0],&m_extdata);

                m_bitrate = bitrate;

                m_aux_sdp_line = NULL;
                m_done_flag = 0;
                m_dummy_rtp_sink = NULL;
            }
            virtual ~h264_session()
            {
                abcdk_h264_extradata_clean(&m_extdata);

                delete[] m_aux_sdp_line;
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
            
            static void afterPlayingDummy(void *clientData)
            {
                h264_session *session_ctx = (h264_session *)clientData;
                session_ctx->afterPlayingDummy1();
            }

            static void checkForAuxSDPLine(void *clientData)
            {
                h264_session *session_ctx = (h264_session *)clientData;
                session_ctx->checkForAuxSDPLine1();
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
                estBitrate = ABCDK_CLAMP(m_bitrate,(unsigned int)1500, m_bitrate); // bps, 1500 ~ MAX.

                h264_source *source_ctx = h264_source::createNew(envir(), codec_id(),m_rgbuf_ctx_p);
                if (!source_ctx)
                    return NULL;

                return H264VideoStreamFramer::createNew(envir(), source_ctx);
            }

            virtual RTPSink *createNewRTPSink(Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource *inputSource)
            {
                if(!m_extdata.sps || !m_extdata.pps)
                    return NULL;

                return H264VideoRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic, m_extdata.sps->pptrs[0], m_extdata.sps->sizes[0], m_extdata.pps->pptrs[0], m_extdata.pps->sizes[0]);
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //_ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

#endif // ABCDK_RTSP_SERVER_H264_SESSION_HXX
