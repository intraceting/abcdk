/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_AAC_SESSION_HXX
#define ABCDK_RTSP_SERVER_AAC_SESSION_HXX

#include "abcdk/util/object.h"
#include "abcdk/util/basecode.h"
#include "abcdk/util/aac.h"
#include "server_aac_source.hxx"
#include "server_session.hxx"

#ifdef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class aac_session : public session
        {
        private:
            rtsp::ringbuf *m_rgbuf_ctx_p;
            abcdk_aac_adts_header_t m_extdata;

        public:
            static aac_session *createNew(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, Boolean reuseFirstSource = True, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
            {
                return new aac_session(env, codec_id, rgbuf_ctx, extdata, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP);
            }

            static void deleteOld(aac_session **ctx)
            {
                aac_session *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                Medium::close(ctx_p->envir(),ctx_p->name());
                //delete ctx_p;
            }

        protected:
            aac_session(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, Boolean reuseFirstSource = True, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
                : session(env, codec_id, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP)
            {
                m_rgbuf_ctx_p = rgbuf_ctx;

                memset(&m_extdata, 0, sizeof(m_extdata));
                abcdk_aac_extradata_deserialize(extdata->pptrs[0], extdata->sizes[0], &m_extdata);
            }
            virtual ~aac_session()
            {
            }

            virtual FramedSource *createNewStreamSource(unsigned clientSessionId, unsigned &estBitrate)
            {
                estBitrate = 96; // 96 kbps.

                aac_source *source_ctx = aac_source::createNew(envir(),codec_id(), m_rgbuf_ctx_p);
                if (!source_ctx)
                    return NULL;

                return source_ctx;
            }

            virtual RTPSink *createNewRTPSink(Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource *inputSource)
            {
                int type = 2; // AAC-LC
                int channels = abcdk_aac_channels(m_extdata.channel_cfg);
                int frequency = abcdk_aac_sample_rates(m_extdata.sample_rate_index);

                uint8_t config1 = (type << 3) | (m_extdata.sample_rate_index >> 1);
                uint8_t config2 = ((m_extdata.sample_rate_index & 1) << 7) | (channels << 3);

                std::array<char, 5> buf = {0};
                snprintf(buf.data(), buf.size(), "%02X%02X", config1, config2);

                return MPEG4GenericRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic, frequency, "audio", "AAC-hbr", buf.data(), channels);
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //_ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

#endif // ABCDK_RTSP_SERVER_AAC_SESSION_HXX
