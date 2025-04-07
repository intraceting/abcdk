/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_HXX
#define ABCDK_RTSP_SERVER_HXX

#include "abcdk/rtsp/live555.h"
#include "abcdk/util/rwlock.h"
#include "server_media.hxx"
#include "rwlock_robot.hxx"

#ifdef _RTSP_SERVER_HH

namespace abcdk
{
    namespace rtsp
    {
        class server : public RTSPServer
        {
        private:
            rtsp::rwlock m_playlist_locker;
            std::map<std::string,rtsp_server::meida*> m_playlist;
            
        public:
            static server *createNew(UsageEnvironment &env, Port ourPort, UserAuthenticationDatabase *authDatabase, unsigned reclamationTestSeconds = 65)
            {
                int ourSocketIPv4 = setUpOurSocket(env, ourPort, AF_INET);
                int ourSocketIPv6 = setUpOurSocket(env, ourPort, AF_INET6);
                if (ourSocketIPv4 < 0 && ourSocketIPv6 < 0)
                    return NULL;

                return new server(env, ourSocketIPv4, ourSocketIPv6, ourPort, authDatabase, reclamationTestSeconds);
            }

            static void deleteOld(server **ctx)
            {
                server *ctx_p;

                if(!ctx || !*ctx)
                    return ;
                
                ctx_p = *ctx;
                *ctx = NULL;

                ctx_p->remove_session_all();
                delete ctx_p;
            }

            void remove_session(const char *name)
            {
                rtsp::rwlock_robot autolock(&m_playlist_locker,1);

                std::map<std::string, rtsp_server::meida *>::iterator it = m_playlist.find(name);
                if (it == m_playlist.end())
                    return;

                deleteServerMediaSession(it->second);
                rtsp_server::meida::deleteOld(&it->second);

                m_playlist.erase(it);
            }

            void remove_session_all()
            {
                rtsp::rwlock_robot autolock(&m_playlist_locker,1);
                
                for(auto &t: m_playlist)
                {
                    deleteServerMediaSession(t.second);
                    rtsp_server::meida::deleteOld(&t.second);
                }

                m_playlist.clear();
            }

        protected:
            server(UsageEnvironment &env, int ourSocketIPv4, int ourSocketIPv6, Port ourPort, UserAuthenticationDatabase *authDatabase, unsigned reclamationTestSeconds)
                : RTSPServer(env, ourSocketIPv4, ourSocketIPv6, ourPort, authDatabase, reclamationTestSeconds)
            {
            }

            virtual ~server()
            {
                cleanup();
            }

        protected:
            virtual void lookupServerMediaSession(char const *streamName, lookupServerMediaSessionCompletionFunc *completionFunc, void *completionClientData, Boolean isFirstLookupInSession)
            {
            }
        };
    } // namespace rtsp
} // namespace abcdk

#endif //_RTSP_SERVER_HH

#endif // ABCDK_RTSP_SERVER_HXX