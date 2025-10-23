/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_OPUS_SESSION_HXX
#define ABCDK_RTSP_SERVER_OPUS_SESSION_HXX

#include "abcdk/util/object.h"
#include "abcdk/util/basecode.h"
#include "server_opus_source.hxx"
#include "server_session.hxx"

#ifdef HAVE_LIVE555

namespace abcdk
{
    namespace rtsp_server
    {
        class opus_session : public session
        {
        private:
            rtsp::ringbuf *m_rgbuf_ctx_p;
            int m_channels;
            int m_sample_rate;
            uint32_t m_bitrate;

        public:
            static opus_session *createNew(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, uint32_t bitrate)
            {
                return new opus_session(env, codec_id, rgbuf_ctx, extdata, bitrate);
            }

            static void deleteOld(opus_session **ctx)
            {
                opus_session *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                Medium::close(ctx_p->envir(),ctx_p->name());
                //delete ctx_p;
            }

        protected:
            opus_session(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx, abcdk_object_t *extdata, uint32_t bitrate)
                : session(env, codec_id)
            {
                m_rgbuf_ctx_p = rgbuf_ctx;

                m_channels = ABCDK_PTR2I32(extdata->pptrs[0],0);
                m_sample_rate = ABCDK_PTR2I32(extdata->pptrs[1],0);

                m_bitrate = bitrate;

            }
            virtual ~opus_session()
            {

            }

            virtual FramedSource *createNewStreamSource(unsigned clientSessionId, unsigned &estBitrate)
            {
                estBitrate = ABCDK_CLAMP(m_bitrate,(unsigned int)64, m_bitrate); // bps, 64 ~ MAX.

                opus_source *source_ctx = opus_source::createNew(envir(),codec_id(), m_rgbuf_ctx_p);
                if (!source_ctx)
                    return NULL;

                return source_ctx;
            }

            virtual RTPSink *createNewRTPSink(Groupsock *rtpGroupsock, unsigned char rtpPayloadTypeIfDynamic, FramedSource *inputSource)
            {
                // OPUS使用固定采样率(48000Hz)。

                return SimpleRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic, m_sample_rate, "audio", "OPUS", m_channels, True, False);
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //HAVE_LIVE555

#endif // ABCDK_RTSP_SERVER_OPUS_SESSION_HXX
