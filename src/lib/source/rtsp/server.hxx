/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_HXX
#define ABCDK_RTSP_SERVER_HXX

#include "abcdk/rtsp/rtsp.h"
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
            rtsp::rwlock m_medialist_locker;
            std::map<int,rtsp_server::media*> m_medialist;
            int m_media_index;

            rtsp::rwlock m_cmdlist_locker;
            std::queue<std::pair<int,std::string>> m_cmdlist;
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

                ctx_p->impl_remove_media_all();
                RTSPServer::close(ctx_p);
            }

            void remove_media_all()
            {
                rtsp::rwlock_robot autolock(&m_cmdlist_locker,1);

                m_cmdlist.push(std::pair<int,std::string>(1,""));

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);
            }

            void remove_media(int media)
            {
                rtsp::rwlock_robot autolock(&m_cmdlist_locker,1);

                m_cmdlist.push(std::pair<int,std::string>(1,std::to_string(media)));

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);
            }

            int media_play(int media)
            {
                rtsp::rwlock_robot autolock(&m_cmdlist_locker,1);

                m_cmdlist.push(std::pair<int,std::string>(2,std::to_string(media)));

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                return 0;
            }

            int create_media(char const *name = NULL, char const *info = NULL, char const *desc = NULL)
            {
                rtsp_server::media *media_ctx;
                int idx;

                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                media_ctx = rtsp_server::media::createNew(envir(), name, info, desc);
                if (!media_ctx)
                    return -1;

                idx = (m_media_index += 1);

                m_medialist[idx] = media_ctx;

                return idx;
            }

            int media_add_stream(int media, int codec, abcdk_object_t *extdata, int cache)
            {
                int idx;

                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                std::map<int, rtsp_server::media *>::iterator it = m_medialist.find(media);
                if (it == m_medialist.end())
                    return -1;

                idx = it->second->add_stream(codec,extdata,cache);
                if(idx <= 0)
                    return -1;

                return idx;
            }

            int media_append_stream(int media, int stream, const void *data, size_t size, int64_t dts, int64_t pts, int64_t dur)
            {
                int chk;

                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                std::map<int, rtsp_server::media *>::iterator it = m_medialist.find(media);
                if (it == m_medialist.end())
                    return -1;

                chk = it->second->append_stream(stream, data, size, dts, pts, dur);
                if(chk != 0)
                    return -1;

                return 0;
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

            static void async_cmd_process(void *clientData)
            {
                server *ctx_p = (server *)clientData;
                std::pair<int, std::string> cmdinfo;

                rtsp::rwlock_robot autolock(&ctx_p->m_cmdlist_locker, 1);

                if(ctx_p->m_cmdlist.size() <= 0)
                    return;

                cmdinfo = ctx_p->m_cmdlist.front();
                ctx_p->m_cmdlist.pop();

                if (cmdinfo.first == 1)
                {
                    if (cmdinfo.second.size() <= 0)
                        ctx_p->impl_remove_media_all();
                    else
                        ctx_p->impl_remove_media(atoi(cmdinfo.second.c_str()));
                }
                else if (cmdinfo.first == 2)
                {
                    ctx_p->impl_media_play(atoi(cmdinfo.second.c_str()));
                }
            }

            void impl_remove_media_all()
            {
                rtsp::rwlock_robot autolock(&m_medialist_locker,1);
                
                for(auto &t: m_medialist)
                {
                    deleteServerMediaSession(t.second);
                    rtsp_server::media::deleteOld(&t.second);
                }

                m_medialist.clear();
            }

            void impl_remove_media(int media)
            {
                rtsp::rwlock_robot autolock(&m_medialist_locker,1);

                std::map<int, rtsp_server::media *>::iterator it = m_medialist.find(media);
                if (it == m_medialist.end())
                    return;

                deleteServerMediaSession(it->second);
                rtsp_server::media::deleteOld(&it->second);

                m_medialist.erase(it);
            }

            int impl_media_play(int media)
            {
                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                std::map<int, rtsp_server::media *>::iterator it = m_medialist.find(media);
                if (it == m_medialist.end())
                    return -1;

                addServerMediaSession(it->second);

                return 0;
            }
        };
    } // namespace rtsp
} // namespace abcdk

#endif //_RTSP_SERVER_HH

#endif // ABCDK_RTSP_SERVER_HXX