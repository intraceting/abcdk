/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_H2645_SOURCE_HXX
#define ABCDK_RTSP_SERVER_H2645_SOURCE_HXX

#include "abcdk/util/h2645.h"
#include "server_general_source.hxx"

#ifdef _FRAMED_SOURCE_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class h2645_source : public general_source
        {
        private:
            rtsp::packet m_cache;
            const void *m_cache_p;
            const void *m_cache_pos;
            const void *m_cache_end;
        public:
            static h2645_source *createNew(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx)
            {
                return new h2645_source(env, codec_id, rgbuf_ctx);
            }

        protected:
            h2645_source(UsageEnvironment &env, int codec_id, rtsp::ringbuf *rgbuf_ctx)
                : general_source(env, codec_id, rgbuf_ctx)
            {
                m_cache_p = m_cache_pos = m_cache_end = NULL; 
            }

            virtual ~h2645_source()
            {
            }

            virtual int fetch(rtsp::packet &pkt)
            {
                uint8_t sc3[3] = {0, 0, 1}, sc4[4] = {0, 0, 0, 1};
                int chk;

                if(!m_cache_pos || m_cache_pos > m_cache_end)
                {
                    chk = general_source::fetch(pkt);
                    if(chk == 0)
                        return 0;

                    /*如果没有起始码则直接返回。*/
                    if (memcmp(pkt.data(), sc4, 4) != 0 && memcmp(pkt.data(), sc3, 3) != 0)
                        return 1;

                    /*复制到缓存。*/
                    m_cache = pkt;
                    pkt.clear();

                    /*存在起始码必须先拆包，因为RTP不支持码流内的拼包。*/
                    m_cache_pos = m_cache.data();
                    m_cache_end = ABCDK_PTR2VPTR(m_cache.data(), m_cache.size() - 1); /*末尾指针要减1。*/
                }
                
                /*拆包，查找包首地址(跳过超始码)。*/
                m_cache_p = abcdk_h2645_packet_split((void **)&m_cache_pos, m_cache_end);
                if (m_cache_p == NULL)
                    return 0;
                
                /*复制。*/
                pkt.copy_from(m_cache_p, (size_t)m_cache_pos - (size_t)m_cache_p, m_cache.dur());
                return 1;
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_FRAMED_SOURCE_HH

#endif // ABCDK_RTSP_SERVER_H2645_SOURCE_HXX