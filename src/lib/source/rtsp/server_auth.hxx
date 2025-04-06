/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_SERVER_AUTH_HXX
#define ABCDK_RTSP_SERVER_AUTH_HXX

#include "abcdk/rtsp/live555.h"
#include "abcdk/util/rwlock.h"

#ifdef _GENERIC_MEDIA_SERVER_HH

namespace abcdk
{
    namespace rtsp_server
    {
        class auth : public UserAuthenticationDatabase
        {
        private:
            std::map<std::string, std::array<char,NAME_MAX>> m_db;//固定大小的值，因为查询接口是用指针，但是对象又随时被删除。
            abcdk_rwlock_t *m_locker;
        public:
            auth(char const *realm = NULL)
                : UserAuthenticationDatabase(realm)
            {
                m_locker = abcdk_rwlock_create();
            }

            virtual ~auth()
            {
                abcdk_rwlock_destroy(&m_locker);
            }

        public:
            void addUserRecord(char const *username, char const *password)
            {
                assert(username != NULL && password != NULL);

                abcdk_rwlock_wrlock(m_locker,1);

                strncpy(m_db[username].data(),password,NAME_MAX);

                abcdk_rwlock_unlock(m_locker);
            }

            void removeUserRecord(char const *username)
            {
                assert(username != NULL);

                abcdk_rwlock_wrlock(m_locker,1);

                std::map<std::string, std::array<char,NAME_MAX>>::iterator it = m_db.find(username);
                if (it == m_db.end())
                    return ;

                m_db[username].fill('\0');

                abcdk_rwlock_unlock(m_locker);
            }

            char const *lookupPassword(char const *username)
            {
                char *p = NULL;

                assert(username != NULL);

                abcdk_rwlock_rdlock(m_locker,1);

                std::map<std::string, std::array<char,NAME_MAX>>::iterator it = m_db.find(username);
                if (it != m_db.end())
                    p = it->second.data();

                abcdk_rwlock_unlock(m_locker);

                return p;
            }
        };
    } // namespace rtsp_server
} // namespace abcdk

#endif //_GENERIC_MEDIA_SERVER_HH

#endif // ABCDK_RTSP_SERVER_AUTH_HXX