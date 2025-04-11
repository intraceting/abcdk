/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_G711_SOURCE_HXX
#define ABCDK_RTSP_SERVER_G711_SOURCE_HXX

#include "server_source.hxx"
#include "ringbuf.hxx"

#ifdef _FRAMED_SOURCE_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class g711_source : public source
        {
        private:
            rtsp::ringbuf *m_rgbuf_ctx_p;
            uint64_t m_rgbuf_idx;

        public:
            static g711_source *createNew(UsageEnvironment &env,int codec_id, rtsp::ringbuf *rgbuf_ctx)
            {
                return new g711_source(env, codec_id, rgbuf_ctx);
            }

        protected:
            g711_source(UsageEnvironment &env,int codec_id, rtsp::ringbuf *rgbuf_ctx)
                : source(env, codec_id)
            {
                m_rgbuf_ctx_p = rgbuf_ctx;
                m_rgbuf_idx = 0;
            }

            virtual ~g711_source()
            {
                
            }

            int fetch(rtsp::packet &pkt)
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

#endif //_FRAMED_SOURCE_HH

#endif // ABCDK_RTSP_SERVER_G711_SOURCE_HXX