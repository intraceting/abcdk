/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_SESSION_HXX
#define ABCDK_RTSP_SERVER_SESSION_HXX

#include "abcdk/rtsp/rtsp.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/socket.h"
#include "packet.hxx"

#ifdef _ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class session : public OnDemandServerMediaSubsession
        {
        private:
            int m_codec_id;

        public:
            int codec_id()
            {
                return m_codec_id;
            }

        protected:
            session(UsageEnvironment &env, int codec_id, Boolean reuseFirstSource = True, portNumBits initialPortNum = 6970, Boolean multiplexRTCPWithRTP = False)
                : OnDemandServerMediaSubsession(env, reuseFirstSource, initialPortNum, multiplexRTCPWithRTP)
            {
                m_codec_id = codec_id;
            }

            virtual ~session()
            {
                
            }

        protected:
            void startStream(unsigned clientSessionId, void *streamToken, TaskFunc *rtcpRRHandler, void *rtcpRRHandlerClientData, unsigned short &rtpSeqNum, unsigned &rtpTimestamp,
                             ServerRequestAlternativeByteHandler *serverRequestAlternativeByteHandler, void *serverRequestAlternativeByteHandlerClientData)
            {

                abcdk_sockaddr_t remote_addr = {0}, local_addr = {0};
                char remote_str[NAME_MAX] = {0}, local_str[NAME_MAX] = {0};

                Destinations *dest_p = (Destinations *)(fDestinationsHashTable->Lookup((char const *)(size_t)clientSessionId));
                if (dest_p)
                {
                    if (dest_p->isTCP && dest_p->tcpSocketNum >= 0)
                    {
                        abcdk_socket_getname(dest_p->tcpSocketNum, &remote_addr, &local_addr);

                        abcdk_sockaddr_to_string(remote_str, &remote_addr, 0);
                        abcdk_sockaddr_to_string(local_str, &local_addr, 0);
                    }
#if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                    else if (dest_p->addr.ss_family == AF_INET6 || dest_p->addr.ss_family == AF_INET)
                    {
                        abcdk_sockaddr_to_string(remote_str, (abcdk_sockaddr_t *)&dest_p->addr, 0);
                    }
#else // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                    else
                    {
                        inet_ntop(AF_INET, &addr->addr4.sin_addr, remote_str, NAME_MAX);
                    }
#endif // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                }

                abcdk_trace_printf(LOG_DEBUG, "++++++++++++++++++++\nFunction: %s\nSession ID: %u\nMedia Name: %s\nTransport: %s\nRemote Address: %s\nLocal Address: %s\n++++++++++++++++++++\n",
                                   __FUNCTION__, clientSessionId, fParentSession->streamName(), (dest_p->isTCP ? "TCP" : "UDP"), remote_str, local_str);

                OnDemandServerMediaSubsession::startStream(clientSessionId, streamToken, rtcpRRHandler, rtcpRRHandlerClientData, rtpSeqNum, rtpTimestamp, serverRequestAlternativeByteHandler, serverRequestAlternativeByteHandlerClientData);
            }

            void deleteStream(unsigned clientSessionId, void *&streamToken)
            {
                abcdk_trace_printf(LOG_DEBUG, "++++++++++++++++++++\nFunction: %s\nSession ID: %u\n++++++++++++++++++++\n",
                                   __FUNCTION__, clientSessionId);

                OnDemandServerMediaSubsession::deleteStream(clientSessionId, streamToken);
            }
        };
    } // namespace rtsp_server
} // namespace abcdk
#endif //_ON_DEMAND_SERVER_MEDIA_SUBSESSION_HH

#endif // ABCDK_RTSP_SERVER_SESSION_HXX
