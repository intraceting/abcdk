/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_MEDIA_HXX
#define ABCDK_RTSP_SERVER_MEDIA_HXX

#include "abcdk/rtsp/rtsp.h"
#include "abcdk/util/rwlock.h"
#include "ringbuf.hxx"
#include "server_h264_session.hxx"
#include "server_h265_session.hxx"
#include "server_acc_session.hxx"

#ifdef _SERVER_MEDIA_SESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class media : public ServerMediaSession
        {
        private:
            std::map<int, rtsp::ringbuf> m_rgbuf;
            int m_stream_index;

        public:
            static media *createNew(UsageEnvironment &env, char const *name = NULL, char const *info = NULL, char const *desc = NULL)
            {
                return new media(env, name, info, desc);
            }

            static void deleteOld(media **ctx)
            {
                media *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                ctx_p->deleteAllSubsessions();
                Medium::close(ctx_p);// 删除对象，防止内存泄漏。
            }

            int add_stream(int codec, abcdk_object_t *extdata, int cache)
            {
                int idx;
                Boolean bchk;

                idx = (m_stream_index += 1);
                m_rgbuf[idx].reset(cache);

                if (codec == ABCDK_RTSP_CODEC_H264)
                    bchk = addSubsession(rtsp_server::h264_session::createNew(envir(), &m_rgbuf[idx], extdata));
                else if (codec == ABCDK_RTSP_CODEC_H265)
                    bchk = addSubsession(rtsp_server::h265_session::createNew(envir(), &m_rgbuf[idx], extdata));
                else
                    bchk = False;

                if (!bchk)
                {
                    m_rgbuf.erase(idx);
                    m_stream_index -= 1;
                    return -1;
                }

                return idx;
            }

            int append_stream(int stream,const void *data, size_t size, int64_t dts, int64_t pts, int64_t dur)
            {
                std::map<int, rtsp::ringbuf>::iterator it = m_rgbuf.find(stream);
                if(it == m_rgbuf.end())
                    return -1;

                it->second.write(data,size,dts,pts,dur);

                return 0;
            }

        protected:
            media(UsageEnvironment &env, char const *name, char const *info, char const *desc)
                : ServerMediaSession(env, name, info, desc, false, NULL)
            {
            }

            virtual ~media()
            {
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_SERVER_MEDIA_SESSION_HH

#endif // ABCDK_RTSP_SERVER_MEDIA_HXX