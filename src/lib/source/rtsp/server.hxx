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
#include "abcdk/util/socket.h"
#include "server_media.hxx"
#include "rwlock_robot.hxx"
#include "waiter.hxx"

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
            
            rtsp::waiter m_cmdlist_waiter;
            rtsp::rwlock m_cmdlist_locker;
            std::queue<std::pair<int,std::vector<std::string>>> m_cmdlist;
        public:
            static server *createNew(UsageEnvironment &env, Port ourPort, UserAuthenticationDatabase *authDatabase, unsigned reclamationTestSeconds = 65)
            {
                int ourSocketIPv4 = setUpOurSocket(env, ourPort, AF_INET);
                int ourSocketIPv6 = setUpOurSocket(env, ourPort, AF_INET6);
                if (ourSocketIPv4 < 0 && ourSocketIPv6 < 0)
                    return NULL;

                if (ourSocketIPv4 >= 0)
                {
                    abcdk_sockopt_option_int_set(ourSocketIPv4, SOL_SOCKET, SO_REUSEPORT, 1);
                    abcdk_sockopt_option_int_set(ourSocketIPv4, SOL_SOCKET, SO_REUSEADDR, 1);
                }

                if (ourSocketIPv6 >= 0)
                {
                    abcdk_sockopt_option_int_set(ourSocketIPv6, SOL_SOCKET, SO_REUSEPORT, 1);
                    abcdk_sockopt_option_int_set(ourSocketIPv6, SOL_SOCKET, SO_REUSEADDR, 1);
                }

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
                delete ctx_p;
            }

            void remove_media_all()
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;

                m_cmdlist_locker.lock();

                std::pair<int,std::vector<std::string>> param;

                param.first = 1;//cmd

                param.second.resize(2);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = "";

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key,15*1000);
                if(!rsp_p)
                    return;
                
                delete rsp_p;
            }

            void remove_media(int media)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = 1; // cmd

                param.second.resize(2);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = std::to_string(media);

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key, 15 * 1000);
                if (!rsp_p)
                    return;

                delete rsp_p;
            }

            int media_play(int media)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;
                int chk;

                m_cmdlist_locker.lock();

                std::pair<int,std::vector<std::string>> param;

                param.first = 2;//cmd

                param.second.resize(2);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = std::to_string(media);

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key,15*1000);
                if(!rsp_p)
                    return -1;

                chk = atoi(rsp_p->c_str());
                delete rsp_p;

                return chk;
            }

            int create_media(char const *name = NULL, char const *info = NULL, char const *desc = NULL)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;
                int idx;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = 3; // cmd

                param.second.resize(4);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = (name ? name : "");
                param.second[2] = (info ? info : "");
                param.second[3] = (desc ? desc : "");

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key, 15 * 1000);
                if (!rsp_p)
                    return -1;

                idx = atoi(rsp_p->c_str());
                delete rsp_p;

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
                OutPacketBuffer::maxSize = 1024 * 1024; // 1MB
            }

            virtual ~server()
            {
                cleanup();
            }

        protected:
            static void async_cmd_process(void *clientData)
            {
                server *ctx_p = (server *)clientData;
                std::pair<int, std::vector<std::string>> cmdinfo;
                uint64_t rsp_key;
                int chk;
                

                /*no command.*/
                cmdinfo.first = 0;

                /*pop from CMDLIST.*/
                ctx_p->m_cmdlist_locker.lock();
                if (ctx_p->m_cmdlist.size() > 0)
                {
                    cmdinfo = ctx_p->m_cmdlist.front();
                    ctx_p->m_cmdlist.pop();
                }
                ctx_p->m_cmdlist_locker.unlock();

                if(cmdinfo.first == 0)
                {
                    return;
                }
                else if (cmdinfo.first == 1)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    if (cmdinfo.second[1].size() <= 0)
                        ctx_p->impl_remove_media_all();
                    else
                        ctx_p->impl_remove_media(atoi(cmdinfo.second[1].c_str()));

                    ctx_p->m_cmdlist_waiter.response(rsp_key,new std::string(""));
                }
                else if (cmdinfo.first == 2)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    chk = ctx_p->impl_media_play(atoi(cmdinfo.second[1].c_str()));

                    ctx_p->m_cmdlist_waiter.response(rsp_key,new std::string(std::to_string(chk)));
                }
                else if (cmdinfo.first == 3)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    chk = ctx_p->impl_create_media(cmdinfo.second[1].c_str(),cmdinfo.second[2].c_str(),cmdinfo.second[3].c_str());

                    ctx_p->m_cmdlist_waiter.response(rsp_key,new std::string(std::to_string(chk)));
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

            int impl_create_media(char const *name = NULL, char const *info = NULL, char const *desc = NULL)
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
        };
    } // namespace rtsp
} // namespace abcdk

#endif //_RTSP_SERVER_HH

#endif // ABCDK_RTSP_SERVER_HXX