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

#ifdef HAVE_LIVE555

namespace abcdk
{
    namespace rtsp_server
    {
        class h265_session : public session
        {
        private:
            rtsp::ringbuf *m_rgbuf_ctx_p;
            abcdk_hevc_extradata_t m_extdata;
            uint32_t m_bitrate;

            char *m_aux_sdp_line;
            char m_done_flag;
            RTPSink *m_dummy_rtp_sink;

        public:
            static h265_session *createNew(UsageEnvironment &env,int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, uint32_t bitrate)
            {
                return new h265_session(env, codec_id, rgbuf_ctx, extdata, bitrate);
            }

            static void deleteOld(h265_session **ctx)
            {
                h265_session *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                Medium::close(ctx_p->envir(),ctx_p->name());
                //delete ctx_p;
            }

        protected:
            h265_session(UsageEnvironment &env,int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, uint32_t bitrate)
                : session(env, codec_id)
            {
                m_rgbuf_ctx_p = rgbuf_ctx;

                memset(&m_extdata,0,sizeof(m_extdata));
                abcdk_hevc_extradata_deserialize(extdata->pptrs[0], extdata->sizes[0],&m_extdata);

                m_bitrate = bitrate;

                m_aux_sdp_line = NULL;
                m_done_flag = 0;
                m_dummy_rtp_sink = NULL;
            }
            virtual ~h265_session()
            {
                abcdk_hevc_extradata_clean(&m_extdata);

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
                h265_session *session_ctx = (h265_session *)clientData;
                session_ctx->afterPlayingDummy1();
            }

            static void checkForAuxSDPLine(void *clientData)
            {
                h265_session *session_ctx = (h265_session *)clientData;
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

                h265_source *source_ctx = h265_source::createNew(envir(), codec_id(), m_rgbuf_ctx_p);
                if (!source_ctx)
                    return NULL;

                return H265VideoStreamFramer::createNew(envir(), source_ctx);
            }

            virtual RTPSink *createNewRTPSink(Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource *inputSource)
            {
                uint8_t *vps_p = NULL, *sps_p = NULL, *pps_p = NULL;
                int vps_size = 0, sps_size = 0, pps_size = 0;

                for (int i = 0; i < m_extdata.nal_array_num; i++)
                {
                    abcdk_object_t *nal_p = m_extdata.nal_array[i].nal;

                    for (int j = 0; j < m_extdata.nal_array[i].nal_num; j++)
                    {
                        uint8_t unit_type = (nal_p->pptrs[j][0] >> 1) & 0x3F;

                        if (unit_type == 32)
                        {
                            vps_p = nal_p->pptrs[j];
                            vps_size = nal_p->sizes[j];
                        }
                        else if (unit_type == 33)
                        {
                            sps_p = nal_p->pptrs[j];
                            sps_size = nal_p->sizes[j];
                        }
                        else if (unit_type == 34)
                        {
                            pps_p = nal_p->pptrs[j];
                            pps_size = nal_p->sizes[j];
                        }
                    }
                }

                if(!vps_p || !sps_p || !pps_p)
                    return NULL;

                return H265VideoRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic, vps_p, vps_size, sps_p, sps_size, pps_p, pps_size);
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //HAVE_LIVE555

#endif // ABCDK_RTSP_SERVER_H265_SESSION_HXX
