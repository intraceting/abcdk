/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_G711_SESSION_HXX
#define ABCDK_RTSP_SERVER_G711_SESSION_HXX

#include "abcdk/util/object.h"
#include "abcdk/util/basecode.h"
#include "server_g711_source.hxx"
#include "server_session.hxx"

#ifdef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class g711_session : public session
        {
        private:
            rtsp::ringbuf *m_rgbuf_ctx_p;
            int m_channels;
            int m_sample_rate;

        public:
            static g711_session *createNew(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, Boolean reuseFirstSource = True, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
            {
                return new g711_session(env, codec_id, rgbuf_ctx, extdata, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP);
            }

            static void deleteOld(g711_session **ctx)
            {
                g711_session *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                Medium::close(ctx_p->envir(),ctx_p->name());
                //delete ctx_p;
            }

        protected:
            g711_session(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, Boolean reuseFirstSource = True, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
                : session(env, codec_id, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP)
            {
                m_rgbuf_ctx_p = rgbuf_ctx;

                m_channels = ABCDK_PTR2I32(extdata->pptrs[0],0);
                m_sample_rate = ABCDK_PTR2I32(extdata->pptrs[1],0);

            }
            virtual ~g711_session()
            {

            }

            virtual FramedSource *createNewStreamSource(unsigned clientSessionId, unsigned &estBitrate)
            {
                estBitrate = 96; // 96 kbps.

                g711_source *source_ctx = g711_source::createNew(envir(),codec_id(), m_rgbuf_ctx_p);
                if (!source_ctx)
                    return NULL;

                return source_ctx;
            }

            virtual RTPSink *createNewRTPSink(Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource *inputSource)
            {
                /*
                G.711 标准 RTP payload 是 固定 8000 Hz / 1 channel，标准 Payload Type 是：
                    0: PCMU (μ-law)
                    8: PCMA (A-law)

                如果非要用 48kHz/双声道 + G.711，那必须用 动态负载类型（如 97）+ SDP 明确声明，且接收方要能处理。
                */

                if (codec_id() == ABCDK_RTSP_CODEC_G711A)
                    return SimpleRTPSink::createNew(envir(), rtpGroupsock, (m_channels > 1 ? rtpPayloadTypeIfDynamic : 8), m_sample_rate, "audio", "PCMA", m_channels, True, False);
                else if (codec_id() == ABCDK_RTSP_CODEC_G711U)
                    return SimpleRTPSink::createNew(envir(), rtpGroupsock, (m_channels > 1 ? rtpPayloadTypeIfDynamic : 0), m_sample_rate, "audio", "PCMU", m_channels, True, False);
                else
                    return NULL;
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //_ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

#endif // ABCDK_RTSP_SERVER_G711_SESSION_HXX
