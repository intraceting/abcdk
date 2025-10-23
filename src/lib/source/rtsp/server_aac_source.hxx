/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_AAC_SOURCE_HXX
#define ABCDK_RTSP_SERVER_AAC_SOURCE_HXX

#include "server_source.hxx"
#include "ringbuf.hxx"

#ifdef HAVE_LIVE555

namespace abcdk
{
    namespace rtsp_server
    {
        class aac_source : public source
        {
        private:
            rtsp::ringbuf *m_rgbuf_ctx_p;
            uint64_t m_rgbuf_idx;

        public:
            static aac_source *createNew(UsageEnvironment &env,int codec_id, rtsp::ringbuf *rgbuf_ctx)
            {
                return new aac_source(env, codec_id, rgbuf_ctx);
            }

        protected:
            aac_source(UsageEnvironment &env,int codec_id, rtsp::ringbuf *rgbuf_ctx)
                : source(env, codec_id)
            {
                m_rgbuf_ctx_p = rgbuf_ctx;
                m_rgbuf_idx = 0;
            }

            virtual ~aac_source()
            {
                
            }

            virtual int fetch(rtsp::packet &pkt)
            {
                int chk;

                chk = m_rgbuf_ctx_p->read(pkt, m_rgbuf_idx);
                if (chk <= 0)
                    return 0;

                return 1;
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //HAVE_LIVE555

#endif // ABCDK_RTSP_SERVER_AAC_SOURCE_HXX