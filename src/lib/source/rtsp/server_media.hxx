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
#include "server_aac_session.hxx"

#ifdef _SERVER_MEDIA_SESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class media : public ServerMediaSession
        {
        private:
            std::map<int, std::pair<rtsp_server::session *, rtsp::ringbuf>> m_stream;
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
                Medium::close(ctx_p);
            }

            int add_stream(int codec, abcdk_object_t *extdata, int cache)
            {
                rtsp_server::session *subsession_ctx;
                int idx;
                Boolean bchk;

                idx = (m_stream_index += 1);
                m_stream[idx].first = NULL;
                m_stream[idx].second.reset(cache);

                if (codec == ABCDK_RTSP_CODEC_H264)
                    subsession_ctx = rtsp_server::h264_session::createNew(envir(),ABCDK_RTSP_CODEC_H264, &m_stream[idx].second, extdata);
                else if (codec == ABCDK_RTSP_CODEC_H265)
                    subsession_ctx = rtsp_server::h265_session::createNew(envir(),ABCDK_RTSP_CODEC_H265, &m_stream[idx].second, extdata);
                else if (codec == ABCDK_RTSP_CODEC_AAC)
                    subsession_ctx = rtsp_server::aac_session::createNew(envir(),ABCDK_RTSP_CODEC_AAC, &m_stream[idx].second, extdata);
                else
                    goto ERR;

                bchk = addSubsession(subsession_ctx);
                if (!bchk)
                {
                    if (codec == ABCDK_RTSP_CODEC_H264)
                        rtsp_server::h264_session::deleteOld((rtsp_server::h264_session **)&subsession_ctx);
                    else if (codec == ABCDK_RTSP_CODEC_H265)
                        rtsp_server::h265_session::deleteOld((rtsp_server::h265_session **)&subsession_ctx);
                    else if (codec == ABCDK_RTSP_CODEC_AAC)
                        rtsp_server::aac_session::deleteOld((rtsp_server::aac_session **)&subsession_ctx);

                    goto ERR;
                }

                /*copy .*/
                m_stream[idx].first = subsession_ctx;

                return idx;

            ERR:

                m_stream_index -= 1;//-1
                m_stream.erase(idx);

                return -1;

            }

            int append_stream(int stream,const void *data, size_t size, int64_t dts, int64_t pts, int64_t dur)
            {
                uint8_t startcode3[3] = {0,0,1};
                uint8_t startcode4[4] = {0,0,0,1};

                std::map<int, std::pair<rtsp_server::session *, rtsp::ringbuf>>::iterator it = m_stream.find(stream);
                if(it == m_stream.end())
                    return -1;

                /*H264，H265不需要起始码。*/
                if (it->second.first->codec_id() == ABCDK_RTSP_CODEC_H264 ||
                    it->second.first->codec_id() == ABCDK_RTSP_CODEC_H265)
                {
                    if (memcmp(data, startcode4, 4) == 0)
                    {
                        data = ABCDK_PTR2VPTR(data, 4);
                        size -= 4;
                    }
                    else if (memcmp(data, startcode4, 3) == 0)
                    {
                        data = ABCDK_PTR2VPTR(data, 3);
                        size -= 3;
                    }
                }

                it->second.second.write(data,size,dts,pts,dur);

                return 0;
            }

        protected:
            media(UsageEnvironment &env, char const *name, char const *info, char const *desc)
                : ServerMediaSession(env, name, info, desc, False, NULL)
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