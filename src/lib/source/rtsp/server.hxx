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
#include "server_auth.hxx"
#include "server_media.hxx"
#include "rwlock_robot.hxx"
#include "waiter.hxx"

#define ABCDK_RTSP_SERVER_REALM "ABCDK MediaServer"

#ifdef _RTSP_SERVER_HH

namespace abcdk
{
    namespace rtsp
    {
        class server : public RTSPServer
        {
        protected:
            enum async_cmd
            {
                MEDIA_DEL = 1,
                MEDIA_PLAY = 2,
                MEDIA_ADD = 3,
                USER_ADD = 4,
                USER_DEL = 5
            };

        private:
            rtsp::rwlock m_medialist_locker;

            /*<name,<flag,ctx*>>*/
            std::map<std::string, std::pair<int, rtsp_server::media *>> m_medialist;

            rtsp::waiter m_cmdlist_waiter;
            rtsp::rwlock m_cmdlist_locker;

            /*<<cmd,<param>>>*/
            std::queue<std::pair<int, std::vector<std::string>>> m_cmdlist;

            /*授权管理。*/
            abcdk::rtsp_server::auth *m_auth_ctx;

        public:
            static server *createNew(UsageEnvironment &env, Port ourPort, int flag, unsigned reclamationTestSeconds = 65)
            {
                int sock_fd = -1, sock6_fd = -1;

#if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200

                if (flag & 0x01)
                    sock_fd = setUpOurSocket(env, ourPort, AF_INET);

                if (flag & 0x02)
                    sock6_fd = setUpOurSocket(env, ourPort, AF_INET6);

                if (sock_fd < 0 && sock6_fd < 0)
                    return NULL;

                /*启用，但可能未创建成功。*/
                if (((flag & 0x01) && sock_fd < 0) || ((flag & 0x02) && sock6_fd < 0))
                {
                    abcdk_closep(&sock_fd);
                    abcdk_closep(&sock6_fd);
                    return NULL;
                }

                return new server(env, sock_fd, sock6_fd, ourPort, reclamationTestSeconds);

#else // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200

                sock_fd = setUpOurSocket(env, ourPort);

                if (sock_fd < 0)
                    return NULL;

                return new server(env, sock_fd, ourPort, reclamationTestSeconds);

#endif // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
            }

            static void deleteOld(server **ctx)
            {
                server *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                Medium::close(ctx_p->envir(), ctx_p->name());
                // delete ctx_p;
            }

            int set_auth(const char *realm)
            {
                abcdk::rtsp_server::auth *auth_old_ctx = NULL;
                abcdk::rtsp_server::auth *auth_new_ctx = NULL;

                /*创建新的。*/
                auth_new_ctx = abcdk::rtsp_server::auth::createNew(realm ? realm : ABCDK_RTSP_SERVER_REALM);
                if (!auth_new_ctx)
                    return -1;

                /*设置新的，返回旧的并释放。*/
                auth_old_ctx = (abcdk::rtsp_server::auth *)setAuthenticationDatabase(m_auth_ctx = auth_new_ctx);
                abcdk::rtsp_server::auth::deleteOld(&auth_old_ctx);

                return 0;
            }

            int set_tls(const char *cert, const char *key, int enable_srtp, int encrypt_srtp)
            {
#if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                setTLSState(cert, key, enable_srtp, encrypt_srtp);
#else //#if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                abcdk_trace_printf(LOG_WARNING, TT("当前Live555版本(%s)暂不支持此功能(%s)。"),LIVEMEDIA_LIBRARY_VERSION_STRING,__FUNCTION__);
#endif //#if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                return 0;
            }

            int add_user(char const *username, char const *password)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;
                int chk;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = async_cmd::USER_ADD; // cmd

                param.second.resize(3);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = username;
                param.second[2] = password;

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key, 5 * 1000);
                if (!rsp_p)
                    return -1;

                chk = atoi(rsp_p->c_str());
                delete rsp_p;

                return chk;
            }

            void remove_user(char const *username)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = async_cmd::USER_DEL; // cmd

                param.second.resize(2);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = username;

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key, 5 * 1000);
                if (!rsp_p)
                    return;

                delete rsp_p;
            }

            void remove_media_all()
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = async_cmd::MEDIA_DEL; // cmd

                param.second.resize(2);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = "";

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key, 5 * 1000);
                if (!rsp_p)
                    return;

                delete rsp_p;
            }

            void remove_media(const char *name)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = async_cmd::MEDIA_DEL; // cmd

                param.second.resize(2);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = name;

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key, 5 * 1000);
                if (!rsp_p)
                    return;

                delete rsp_p;
            }

            int play_media(const char *name)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;
                int chk;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = async_cmd::MEDIA_PLAY; // cmd

                param.second.resize(2);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = name;

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key, 5 * 1000);
                if (!rsp_p)
                    return -1;

                chk = atoi(rsp_p->c_str());
                delete rsp_p;

                return chk;
            }

            int create_media(char const *name, char const *info = NULL, char const *desc = NULL)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;
                int chk;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = async_cmd::MEDIA_ADD; // cmd

                param.second.resize(4);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = name;
                param.second[2] = (info ? info : ABCDK_RTSP_SERVER_REALM);
                param.second[3] = (desc ? desc : ABCDK_RTSP_SERVER_REALM);

                m_cmdlist.push(param);

                envir().taskScheduler().scheduleDelayedTask(0, async_cmd_process, this);

                m_cmdlist_locker.unlock();

                rsp_p = (std::string *)m_cmdlist_waiter.wait(rsp_key, 5 * 1000);
                if (!rsp_p)
                    return -1;

                chk = atoi(rsp_p->c_str());
                delete rsp_p;

                return chk;
            }

            int add_stream(const char *name, int codec, abcdk_object_t *extdata, uint32_t bitrate, int cache)
            {
                int idx;

                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                std::map<std::string, std::pair<int, rtsp_server::media *>>::iterator it = m_medialist.find(name);
                if (it == m_medialist.end())
                    return -1;

                if (it->second.first != 0)
                {
                    abcdk_trace_printf(LOG_WARNING, TT("媒体(%s)已经播放，不能添加新的流。"),name);
                    return -1;
                }

                abcdk_trace_printf(LOG_INFO, TT("媒体(%s)添加新的流(CODEC=%d)。"), name, codec);

                idx = it->second.second->add_stream(codec, extdata, bitrate, cache);
                if (idx <= 0)
                    return -1;

                return idx;
            }

            int play_stream(const char *name, int stream, const void *data, size_t size, int64_t pts, int64_t dur)
            {
                int chk;

                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                std::map<std::string, std::pair<int, rtsp_server::media *>>::iterator it = m_medialist.find(name);
                if (it == m_medialist.end())
                    return -1;

                if (it->second.first == 0)
                {
                    abcdk_trace_printf(LOG_WARNING, TT("媒体尚未播放，不能接收数据包。"));
                    return -1;
                }

                chk = it->second.second->append_stream(stream, data, size, pts, dur);
                if (chk != 0)
                    return -1;

                return 0;
            }

        protected:
            server(UsageEnvironment &env, int ourSocketIPv4, int ourSocketIPv6, Port ourPort, unsigned reclamationTestSeconds)
                : RTSPServer(env, ourSocketIPv4, ourSocketIPv6, ourPort, NULL, reclamationTestSeconds)
            {
                OutPacketBuffer::maxSize = 4 * 1024 * 1024; // 4MB

                m_auth_ctx = NULL; // no auth.
            }

            virtual ~server()
            {
                abcdk::rtsp_server::auth::deleteOld(&m_auth_ctx);
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

                if (cmdinfo.first == 0)
                {
                    return;
                }
                else if (cmdinfo.first == async_cmd::MEDIA_DEL)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    if (cmdinfo.second[1].size() <= 0)
                        ctx_p->impl_remove_media_all();
                    else
                        ctx_p->impl_remove_media(cmdinfo.second[1].c_str());

                    ctx_p->m_cmdlist_waiter.response(rsp_key, new std::string(""));
                }
                else if (cmdinfo.first == async_cmd::MEDIA_PLAY)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    chk = ctx_p->impl_play_media(cmdinfo.second[1].c_str());

                    ctx_p->m_cmdlist_waiter.response(rsp_key, new std::string(std::to_string(chk)));
                }
                else if (cmdinfo.first == async_cmd::MEDIA_ADD)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    chk = ctx_p->impl_create_media(cmdinfo.second[1].c_str(), cmdinfo.second[2].c_str(), cmdinfo.second[3].c_str());

                    ctx_p->m_cmdlist_waiter.response(rsp_key, new std::string(std::to_string(chk)));
                }
                else if (cmdinfo.first == async_cmd::USER_ADD)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    chk = ctx_p->impl_add_user(cmdinfo.second[1].c_str(), cmdinfo.second[2].c_str());

                    ctx_p->m_cmdlist_waiter.response(rsp_key, new std::string(std::to_string(chk)));
                }
                else if (cmdinfo.first == async_cmd::USER_DEL)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    ctx_p->impl_remove_user(cmdinfo.second[1].c_str());

                    ctx_p->m_cmdlist_waiter.response(rsp_key, new std::string(""));
                }
            }

            int impl_add_user(char const *username, char const *password)
            {
                if (!m_auth_ctx)
                    return -1;

                m_auth_ctx->addUserRecord(username, password);

                return 0;
            }

            void impl_remove_user(char const *username)
            {
                if (!m_auth_ctx)
                    return;

                m_auth_ctx->removeUserRecord(username);
            }

            void impl_remove_media_all()
            {
                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                for (auto &t : m_medialist)
                {
                    if (t.second.first)
                        deleteServerMediaSession(t.second.second); // 删除已播放。
                    else
                        rtsp_server::media::deleteOld(&t.second.second); // 删除未播放。
                }

                m_medialist.clear();
            }

            void impl_remove_media(const char *name)
            {
                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                std::map<std::string, std::pair<int, rtsp_server::media *>>::iterator it = m_medialist.find(name);
                if (it == m_medialist.end())
                    return;

                abcdk_trace_printf(LOG_INFO, TT("删除媒体(%s)。"), name);

                if (it->second.first)
                    deleteServerMediaSession(it->second.second); // 删除已播放。
                else
                    rtsp_server::media::deleteOld(&it->second.second); // 删除未播放。

                m_medialist.erase(it);
                
            }

            int impl_play_media(const char *name)
            {
                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                std::map<std::string, std::pair<int, rtsp_server::media *>>::iterator it = m_medialist.find(name);
                if (it == m_medialist.end())
                    return -1;

                /*可能还没有流。*/
                if (it->second.second->stream_count() == 0)
                    return -1;

                /*重复操作直接返回。*/
                if (it->second.first != 0)
                    return 0;

                abcdk_trace_printf(LOG_INFO, TT("播放媒体(%s)。"), name);

                addServerMediaSession(it->second.second);
                it->second.first = 1; // 已播放。

                return 0;
            }

            int impl_create_media(char const *name, char const *info = NULL, char const *desc = NULL)
            {
                rtsp_server::media *media_ctx;

                rtsp::rwlock_robot autolock(&m_medialist_locker, 1);

                std::map<std::string, std::pair<int, rtsp_server::media *>>::iterator it = m_medialist.find(name);
                if (it != m_medialist.end())
                    return -1;

                abcdk_trace_printf(LOG_INFO, TT("创建媒体(%s)。"), name);

                media_ctx = rtsp_server::media::createNew(envir(), name, info, desc);
                if (!media_ctx)
                    return -1;

                m_medialist[name].first = 0; // 未播放。
                m_medialist[name].second = media_ctx;

                return 0;
            }
        };
    } // namespace rtsp
} // namespace abcdk

#endif //_RTSP_SERVER_HH

#endif // ABCDK_RTSP_SERVER_HXX