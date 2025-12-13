/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_RTSP_PAWD_HXX
#define ABCDK_RTSP_PAWD_HXX

#include "abcdk/util/time.h"
#include "abcdk/openssl/totp.h"
#include "abcdk/rtsp/rtsp.h"

namespace abcdk
{
    namespace rtsp
    {
        class pawd
        {
        private:
            std::array<char, NAME_MAX> m_base_data;

            int m_scheme;

            int m_totp_time_step;
            int m_totp_digit_size;

            uint64_t m_totp_counter;
            std::array<char, NAME_MAX> m_totp_data;

        public:
            pawd()
            {
                clear();
            }
            virtual ~pawd()
            {
            }

        public:
            pawd &operator=(const pawd &src)
            {
                /*如是自已, 则忽略.*/
                if (this == &src)
                    return *this;

                m_scheme = src.m_scheme;
                m_base_data = src.m_base_data;
                m_totp_time_step = src.m_totp_time_step;
                m_totp_digit_size = src.m_totp_digit_size;
                m_totp_counter = src.m_totp_counter;
                m_totp_data = src.m_totp_data;

                return *this;
            }

        public:
            void clear()
            {
                m_base_data.fill('\0');
                m_scheme = ABCDK_RTSP_AUTH_NONE;
                m_totp_time_step = 30;
                m_totp_digit_size = 6;
                m_totp_counter = UINT64_MAX;
                m_totp_data.fill('\0');
            }

            void setup(const char *base_data, int scheme, int totp_time_step, int totp_digit_size)
            {
                memcpy(m_base_data.data(), base_data, m_base_data.size());

                m_scheme = scheme;

                m_totp_time_step = totp_time_step;
                m_totp_digit_size = totp_digit_size;
            }

            const char *get()
            {
                if (m_scheme == ABCDK_RTSP_AUTH_NORMAL)
                {
                    return m_base_data.data();
                }

                else if (m_scheme == ABCDK_RTSP_AUTH_TOTP_SHA128 ||
                         m_scheme == ABCDK_RTSP_AUTH_TOTP_SHA256 ||
                         m_scheme == ABCDK_RTSP_AUTH_TOTP_SHA512)
                {
#ifdef OPENSSL_VERSION_NUMBER
                    uint64_t new_counter = abcdk_time_realtime(0) / m_totp_time_step;
                    uint32_t new_num = 0;

                    if (new_counter != m_totp_counter)
                    {
                        if (m_scheme == ABCDK_RTSP_AUTH_TOTP_SHA128)
                            new_num = abcdk_openssl_totp_generate_sha1((uint8_t*)m_base_data.data(), strlen(m_base_data.data()), new_counter);
                        else if (m_scheme == ABCDK_RTSP_AUTH_TOTP_SHA256)
                            new_num = abcdk_openssl_totp_generate_sha256((uint8_t*)m_base_data.data(), strlen(m_base_data.data()), new_counter);
                        else if (m_scheme == ABCDK_RTSP_AUTH_TOTP_SHA512)
                            new_num = abcdk_openssl_totp_generate_sha512((uint8_t*)m_base_data.data(), strlen(m_base_data.data()), new_counter);
                        else
                            assert(0);

                        m_totp_counter = new_counter;

                        snprintf(m_totp_data.data(), m_totp_data.size(), "%0*u", m_totp_digit_size, new_num % (int)pow(10, m_totp_digit_size));
                    }

                    return m_totp_data.data();
#else //#ifdef OPENSSL_VERSION_NUMBER
                    return NULL;
#endif //#ifdef OPENSSL_VERSION_NUMBER
                }

                return NULL;
            }
        };
    } // namespace rtsp
} // namespace abcdk

#endif // ABCDK_RTSP_PAWD_HXX