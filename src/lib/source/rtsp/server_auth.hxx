/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_AUTH_HXX
#define ABCDK_RTSP_SERVER_AUTH_HXX

#include "abcdk/rtsp/live555.h"
#include "rwlock_robot.hxx"

#ifdef _GENERIC_MEDIA_SERVER_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class auth : public UserAuthenticationDatabase
        {
        private:
            rtsp::rwlock m_db_locker;
            std::map<std::string, std::array<char,NAME_MAX>> m_db;//固定大小的值，因为查询接口是用指针，但是对象又随时被删除。
            
        public:
            auth(char const *realm = NULL)
                : UserAuthenticationDatabase(realm)
            {

            }

            virtual ~auth()
            {

            }

        public:
            void addUserRecord(char const *username, char const *password)
            {
                assert(username != NULL && password != NULL);

                rtsp::rwlock_robot autolock(&m_db_locker,1);

                strncpy(m_db[username].data(),password,NAME_MAX);
            }

            void removeUserRecord(char const *username)
            {
                assert(username != NULL);

                rtsp::rwlock_robot autolock(&m_db_locker,1);

                std::map<std::string, std::array<char,NAME_MAX>>::iterator it = m_db.find(username);
                if (it == m_db.end())
                    return ;

                m_db[username].fill('\0');


            }

            char const *lookupPassword(char const *username)
            {
                char *p = NULL;

                assert(username != NULL);

                rtsp::rwlock_robot autolock(&m_db_locker,0);

                std::map<std::string, std::array<char,NAME_MAX>>::iterator it = m_db.find(username);
                if (it != m_db.end())
                    p = it->second.data();

                return p;
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_GENERIC_MEDIA_SERVER_HH

#endif // ABCDK_RTSP_SERVER_AUTH_HXX