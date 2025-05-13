/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_AUTH_HXX
#define ABCDK_RTSP_SERVER_AUTH_HXX

#include "abcdk/openssl/totp.h"
#include "abcdk/rtsp/rtsp.h"
#include "rwlock_robot.hxx"
#include "pawd.hxx"

#ifdef _GENERIC_MEDIA_SERVER_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class auth : public UserAuthenticationDatabase
        {
        private:
            rtsp::rwlock m_db_locker;
            std::map<std::string, rtsp::pawd> m_db;//固定大小的值，因为查询接口是用指针，但是对象又随时可能被删除。
        public:
            static auth *createNew(char const *realm = NULL)
            {
                return new auth(realm);
            }

            static void deleteOld(auth **ctx)
            {
                auth *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                delete ctx_p;
            }

        protected:
            auth(char const *realm = NULL)
                : UserAuthenticationDatabase(realm ? realm : ABCDK_RTSP_SERVER_REALM)
            {

            }

            virtual ~auth()
            {

            }

        public:
            void addUserRecord(char const *username, char const *password, int scheme, int totp_time_step, int totp_digit_size)
            {
                assert(username != NULL && password != NULL);

                rtsp::rwlock_robot autolock(&m_db_locker,1);

                m_db[username].setup(password,scheme,totp_time_step,totp_digit_size);
            }

            void removeUserRecord(char const *username)
            {
                assert(username != NULL);

                rtsp::rwlock_robot autolock(&m_db_locker,1);

                std::map<std::string, rtsp::pawd>::iterator it = m_db.find(username);
                if (it == m_db.end())
                    return ;

                m_db[username].clear();
            }

            char const *lookupPassword(char const *username)
            {
                const char *p = NULL;

                assert(username != NULL);

                rtsp::rwlock_robot autolock(&m_db_locker,0);

                std::map<std::string, rtsp::pawd>::iterator it = m_db.find(username);
                if (it != m_db.end())
                {
                    p = it->second.get();
                    if(*p == '\0')
                        return NULL;
                }

                return p;
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_GENERIC_MEDIA_SERVER_HH

#endif // ABCDK_RTSP_SERVER_AUTH_HXX