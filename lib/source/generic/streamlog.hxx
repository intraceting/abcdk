/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_GENERIC_STREAMLOG_HXX
#define ABCDK_GENERIC_STREAMLOG_HXX

#include "abcdk/util/trace.h"

#include <streambuf>
#include <iostream>
#include <vector>

namespace abcdk
{
    namespace generic
    {
        class streamlog : public std::streambuf
        {
        private:
            std::vector<char> m_buf;
            size_t m_pos;

        public:
            streamlog()
            {
                m_buf.resize(8000);
                m_pos = 0;
            }

            virtual ~streamlog()
            {
                Flush();
            }

        protected:
            virtual int Flush()
            {
                if (m_pos > 0)
                    abcdk_trace_printf(LOG_INFO, "%*s", m_pos, m_buf.data());

                memset(m_buf.data(), 0, m_pos);
                m_pos = 0;

                return 0;
            }

            virtual int_type overflow(int_type c)
            {
                xsputn((char *)&c, 1);

                return c;
            }

            virtual std::streamsize xsputn(char *ptr, std::streamsize s)
            {
                int wlen = snprintf(m_buf.data() + m_pos, m_buf.size() - m_pos, "%*s", (int)s, ptr);
                if (wlen > 0)
                    m_pos += wlen;

                if (m_pos >= m_buf.size() || m_buf.at(m_pos - 1) == '\n')
                    Flush();

                return wlen;
            }
        };
    } //    namespace generic
} // namespace abcdk

#endif // ABCDK_GENERIC_STREAMLOG_HXX