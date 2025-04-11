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
#include "abcdk/util/h2645.h"
#include "ringbuf.hxx"
#include "server_h264_session.hxx"
#include "server_h265_session.hxx"
#include "server_aac_session.hxx"
#include "server_g711_session.hxx"

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
                Medium::close(ctx_p->envir(),ctx_p->name());
                //delete ctx_p;
            }

            int add_stream(int codec, abcdk_object_t *extdata, uint32_t bitrate, uint32_t cache)
            {
                rtsp_server::session *subsession_ctx;
                int idx;
                Boolean bchk;

                idx = (m_stream_index += 1);
                m_stream[idx].first = NULL;
                m_stream[idx].second.reset(cache);

                if (codec == ABCDK_RTSP_CODEC_H264)
                    subsession_ctx = rtsp_server::h264_session::createNew(envir(), ABCDK_RTSP_CODEC_H264, &m_stream[idx].second, extdata, bitrate);
                else if (codec == ABCDK_RTSP_CODEC_H265)
                    subsession_ctx = rtsp_server::h265_session::createNew(envir(), ABCDK_RTSP_CODEC_H265, &m_stream[idx].second, extdata, bitrate);
                else if (codec == ABCDK_RTSP_CODEC_AAC)
                    subsession_ctx = rtsp_server::aac_session::createNew(envir(), ABCDK_RTSP_CODEC_AAC, &m_stream[idx].second, extdata, bitrate);
                else if (codec == ABCDK_RTSP_CODEC_G711A || codec == ABCDK_RTSP_CODEC_G711U)
                    subsession_ctx = rtsp_server::g711_session::createNew(envir(), codec, &m_stream[idx].second, extdata, bitrate);
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
                    else if (codec == ABCDK_RTSP_CODEC_G711A || codec == ABCDK_RTSP_CODEC_G711U)
                        rtsp_server::g711_session::deleteOld((rtsp_server::g711_session **)&subsession_ctx);

                    /**/
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

            int append_stream0(int stream,const void *data, size_t size, int64_t dur)
            {
                uint8_t sc3[3] = {0,0,1},sc4[4] = {0,0,0,1};

                std::map<int, std::pair<rtsp_server::session *, rtsp::ringbuf>>::iterator it = m_stream.find(stream);
                if(it == m_stream.end())
                    return -1;

                /*H264，H265不需要起始码。*/
                if (it->second.first->codec_id() == ABCDK_RTSP_CODEC_H264 ||
                    it->second.first->codec_id() == ABCDK_RTSP_CODEC_H265)
                {
                    if (memcmp(data, sc4, 4) == 0)
                    {
                        data = ABCDK_PTR2VPTR(data, 4);
                        size -= 4;
                    }
                    else if (memcmp(data, sc3, 3) == 0)
                    {
                        data = ABCDK_PTR2VPTR(data, 3);
                        size -= 3;
                    }
                }

                it->second.second.write(data,size,dur);

                return 0;
            }

            int append_stream(int stream, const void *data, size_t size, int64_t dur)
            {
                uint8_t sc3[3] = {0,0,1},sc4[4] = {0,0,0,1};
                const void *p1 = NULL, *p2 = NULL, *p3 = NULL;
                int chk;

                std::map<int, std::pair<rtsp_server::session *, rtsp::ringbuf>>::iterator it = m_stream.find(stream);
                if (it == m_stream.end())
                    return -1;

                /*H264，H265。*/
                if (it->second.first->codec_id() == ABCDK_RTSP_CODEC_H264 ||
                    it->second.first->codec_id() == ABCDK_RTSP_CODEC_H265)
                {
                    if (memcmp(data, sc4, 4) == 0 || memcmp(data, sc3, 3) == 0)
                    {
                        /*存在起始码必须先拆包，因为RTP不支持码流内的拼包。*/
                        p1 = data;
                        p2 = NULL;
                        p3 = ABCDK_PTR2VPTR(data, size - 1); /*末尾指针要减1。*/

                        for (;;)
                        {
                            if (p1 > p3)
                                return 0;

                            p2 = abcdk_h2645_packet_split((void **)&p1, p3);
                            if (p2 == NULL)
                                return 0;

                            chk = append_stream0(stream, p2, (size_t)p1 - (size_t)p2, dur);
                            if (chk != 0)
                                return chk;
                        }
                    }
                    else
                    {
                        return append_stream0(stream, data, size, dur);
                    }
                }
                else
                {
                    return append_stream0(stream, data, size, dur);
                }

                return -1;
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