/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_H2645_SOURCE_HXX
#define ABCDK_RTSP_SERVER_H2645_SOURCE_HXX

#include "server_general_source.hxx"

#ifdef _FRAMED_SOURCE_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class h2645_source : public general_source
        {
        private:

        public:
            static h2645_source *createNew(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx)
            {
                return new h2645_source(env, codec_id, rgbuf_ctx);
            }

        protected:
            h2645_source(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx)
                : general_source(env, codec_id,rgbuf_ctx)
            {

            }

            virtual ~h2645_source()
            {
            }

            virtual int fetch(rtsp::packet &pkt)
            {
                return general_source::fetch(pkt);
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_FRAMED_SOURCE_HH

#endif // ABCDK_RTSP_SERVER_H2645_SOURCE_HXX