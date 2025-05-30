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

            /*启用UDP.*/
            Boolean m_use_udp;

            /*授权管理。*/
            abcdk::rtsp_server::auth *m_auth_ctx;

            /*启用TLS.*/
            Boolean m_use_tls;

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

                return new server(env, sock_fd, sock6_fd, ourPort, (flag & 0x10), reclamationTestSeconds);

#else // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200

                sock_fd = setUpOurSocket(env, ourPort);

                if (sock_fd < 0)
                    return NULL;

                return new server(env, sock_fd, -1, ourPort, reclamationTestSeconds);

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
                auth_new_ctx = abcdk::rtsp_server::auth::createNew(realm);
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
                m_use_tls = True; //
#else                             // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                abcdk_trace_printf(LOG_WARNING, TT("当前Live555版本(%s)暂不支持此功能(%s)。"), LIVEMEDIA_LIBRARY_VERSION_STRING, __FUNCTION__);
#endif                            // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                return 0;
            }

            int add_user(char const *username, char const *password, int scheme, int totp_time_step, int totp_digit_size)
            {
                uint64_t rsp_key = m_cmdlist_waiter.reg();
                std::string *rsp_p = NULL;
                int chk;

                m_cmdlist_locker.lock();

                std::pair<int, std::vector<std::string>> param;

                param.first = async_cmd::USER_ADD; // cmd

                param.second.resize(6);
                param.second[0] = std::to_string(rsp_key);
                param.second[1] = username;
                param.second[2] = password;
                param.second[3] = std::to_string(scheme);
                param.second[4] = std::to_string(totp_time_step);
                param.second[5] = std::to_string(totp_digit_size);

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
                    abcdk_trace_printf(LOG_WARNING, TT("媒体(%s)已经播放，不能添加新的流。"), name);
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
            server(UsageEnvironment &env, int ourSocketIPv4, int ourSocketIPv6, Port ourPort, Boolean use_udp, unsigned reclamationTestSeconds)
#if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                : RTSPServer(env, ourSocketIPv4, ourSocketIPv6, ourPort, NULL, reclamationTestSeconds)
#else  // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
                : RTSPServer(env, ourSocketIPv4, ourPort, NULL, reclamationTestSeconds)
#endif // #if LIVEMEDIA_LIBRARY_VERSION_INT >= 1687219200
            {
                OutPacketBuffer::maxSize = 4 * 1024 * 1024; // 4MB

                m_use_udp = use_udp;
                m_auth_ctx = NULL; // no auth.
                m_use_tls = False;
            }

            virtual ~server()
            {
                abcdk::rtsp_server::auth::deleteOld(&m_auth_ctx);
            }

        protected:
            class CustomRTSPClientConnection : public RTSPServer::RTSPClientConnection
            {
            private:
                std::vector<char> m_re_addr_str;
                std::vector<char> m_lo_addr_str;

            protected:
                friend class server;
                friend class CustomRTSPClientSession;

                CustomRTSPClientConnection(RTSPServer &ourServer, int clientSocket, struct sockaddr_storage const &clientAddr, Boolean useTLS = False)
                    : RTSPServer::RTSPClientConnection(ourServer, clientSocket, clientAddr, useTLS)
                {
                    abcdk_sockaddr_t remote_addr = {0}, local_addr = {0};

                    abcdk_socket_getname(clientSocket, &remote_addr, &local_addr);

                    m_re_addr_str.resize(NAME_MAX);
                    m_lo_addr_str.resize(NAME_MAX);

                    if (remote_addr.family == AF_INET6 || remote_addr.family == AF_INET)
                        abcdk_sockaddr_to_string(m_re_addr_str.data(), &remote_addr, 0);

                    if (local_addr.family == AF_INET6 || local_addr.family == AF_INET)
                        abcdk_sockaddr_to_string(m_lo_addr_str.data(), &local_addr, 0);

                    abcdk_trace_printf(LOG_INFO, "++++++++++online++++++++++\nRemote Address: %s\nLocal Address: %s\n+++++++++online+++++++++++\n",
                                       m_re_addr_str.data(), m_lo_addr_str.data());
                }

                virtual ~CustomRTSPClientConnection()
                {
                    abcdk_trace_printf(LOG_INFO, "++++++++++offline++++++++++\nRemote Address: %s\nLocal Address: %s\n+++++++++offline+++++++++++\n",
                                       m_re_addr_str.data(), m_lo_addr_str.data());
                }
            };

            class CustomRTSPClientSession : public RTSPServer::RTSPClientSession
            {
            private:
                u_int32_t m_sessionId_copy;
                Boolean m_use_udp;

            protected:
                friend class server;

                CustomRTSPClientSession(RTSPServer &ourServer, u_int32_t sessionId,Boolean use_udp = True)
                    : RTSPServer::RTSPClientSession(ourServer, sessionId)
                {
                    m_sessionId_copy = sessionId;
                    m_use_udp = use_udp;
                }

                virtual ~CustomRTSPClientSession()
                {
                }

                void handleCmd_SETUP(RTSPClientConnection *ourClientConnection, char const *urlPreSuffix, char const *urlSuffix, char const *fullRequestStr)
                {
                    // 转自定义类指针。
                    CustomRTSPClientConnection *cu_co_p = (CustomRTSPClientConnection *)ourClientConnection;

                    // abcdk_trace_printf(LOG_DEBUG,"%s",fullRequestStr);

                    // 查找"Transport:"关键字。
                    const char *transport_p = abcdk_strstr_eod(fullRequestStr, "Transport: ", 0);
                    transport_p = (transport_p ? transport_p : abcdk_strstr_eod(fullRequestStr, "Transport:", 0));
                    const char *transport_eod_p = (transport_p != NULL ? abcdk_streod(transport_p, "\r\n") : NULL);

                    // 计算长度。
                    int transport_len = (int)(transport_eod_p ? transport_eod_p - transport_p : 0);

                    abcdk_trace_printf(LOG_INFO, "++++++++++SETUP++++++++++\nSessionId: %u\nPath: %s/%s\nTransport: '%.*s'\nRemoteAddr: %s\nLocalAddr: %s\n++++++++++SETUP++++++++++\n",
                                       m_sessionId_copy, urlPreSuffix, urlSuffix, transport_len, transport_p, cu_co_p->m_re_addr_str.data(), cu_co_p->m_lo_addr_str.data());

                    // 查找"RTP/"关键字。
                    const char *specifier_p = (transport_p != NULL ? abcdk_strstr(transport_p, "RTP/", 1) : NULL);
                    const char *specifier_eod_p = (specifier_p != NULL ? abcdk_streod(specifier_p, ";") : NULL);

                    // 计算长度。
                    int specifier_len = (int)(specifier_eod_p ? specifier_eod_p - specifier_p : 0);

                    // 比较"RTP/AVP/TCP"关键字。
                    Boolean chk_avp_tcp = (specifier_p != NULL ? abcdk_strncmp(specifier_p, "RTP/AVP/TCP", specifier_len, 1) == 0 : False);

                    // 比较"RTP/SAVP/TCP"关键字。
                    Boolean chk_savp_tcp = (specifier_p != NULL ? abcdk_strncmp(specifier_p, "RTP/SAVP/TCP", specifier_len, 1) == 0 : True);

                    if (m_use_udp || chk_avp_tcp || chk_savp_tcp)
                    {
                        // 调用基类，继续处理请求。
                        RTSPServer::RTSPClientSession::handleCmd_SETUP(ourClientConnection, urlPreSuffix, urlSuffix, fullRequestStr);
                    }
                    else
                    {
                        /*通知客户端不支持。*/
                        RTSPServer::RTSPClientSession::setRTSPResponse(ourClientConnection, "461 Unsupported Transport");

                        abcdk_trace_printf(LOG_INFO, "++++++++++SETUP++++++++++\nSessionId: %u\nTransport: '%.*s'\nResponse: '461 Unsupported Transport'\n++++++++++SETUP++++++++++\n",
                                           m_sessionId_copy, transport_len, transport_p);
                    }
                }
            };

        protected:
            RTSPServer::ClientConnection *createNewClientConnection(int clientSocket, struct sockaddr_storage const &clientAddr)
            {
                return new CustomRTSPClientConnection(*this, clientSocket, clientAddr, m_use_tls);
            }

            RTSPServer::ClientSession *createNewClientSession(u_int32_t sessionId)
            {
                return new CustomRTSPClientSession(*this, sessionId, m_use_udp);
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

                    chk = ctx_p->impl_add_user(cmdinfo.second[1].c_str(), cmdinfo.second[2].c_str(), cmdinfo.second[3].c_str(), cmdinfo.second[4].c_str(), cmdinfo.second[5].c_str());

                    ctx_p->m_cmdlist_waiter.response(rsp_key, new std::string(std::to_string(chk)));
                }
                else if (cmdinfo.first == async_cmd::USER_DEL)
                {
                    rsp_key = atoi(cmdinfo.second[0].c_str());

                    ctx_p->impl_remove_user(cmdinfo.second[1].c_str());

                    ctx_p->m_cmdlist_waiter.response(rsp_key, new std::string(""));
                }
            }

            int impl_add_user(char const *username, char const *password, const char *scheme, const char *totp_time_step, const char *totp_digit_size)
            {
                if (!m_auth_ctx)
                    return -1;

                m_auth_ctx->addUserRecord(username, password, atoi(scheme), atoi(totp_time_step), atoi(totp_digit_size));

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