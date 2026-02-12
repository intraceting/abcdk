/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_UTILEX_HXX
#define ABCDK_COMMON_UTILEX_HXX

#include "abcdk.h"

namespace abcdk
{
    namespace common
    {
        namespace UtilEx
        {
            /**
             * 来自于:https://github.com/ggml-org/llama.cpp/blob/master/src/llama-impl.cpp
             * 注:略有修改.
             */
            static inline void string_replace(std::string &s, const std::string &search, const std::string &replace)
            {
                if (search.empty())
                {
                    return;
                }
                std::string builder;
                builder.reserve(s.length());
                size_t pos = 0;
                size_t last_pos = 0;
                while ((pos = s.find(search, last_pos)) != std::string::npos)
                {
                    builder.append(s, last_pos, pos - last_pos);
                    builder.append(replace);
                    last_pos = pos + search.length();
                }
                builder.append(s, last_pos, std::string::npos);
                s = std::move(builder);
            }

            /**
             * 来自于:https://github.com/ggml-org/llama.cpp/blob/master/src/llama-impl.cpp
             * 注:略有修改.
             */
            static inline std::string string_format(const char *fmt, ...)
            {
                va_list ap;
                va_list ap2;
                va_start(ap, fmt);
                va_copy(ap2, ap);
                int size = vsnprintf(NULL, 0, fmt, ap);
                // GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
                std::vector<char> buf(size + 1);
                int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
                // GGML_ASSERT(size2 == size);
                va_end(ap2);
                va_end(ap);
                return std::string(buf.data(), size);
            }

            static inline pid_t popen(const char *uid, const char *gid, const char *envs,
                                      const char *rpath, const char *wpath, const char *cmd,
                                      int *stdin_fd = NULL, int *stdout_fd = NULL, int *stderr_fd = NULL)
            {
                uid_t u_id = ((uid && uid[0]) ? atoi(uid) : UINT32_MAX);
                gid_t g_id = ((gid && gid[0]) ? atoi(gid) : UINT32_MAX);

                std::shared_ptr<abcdk_object_t> obj_envs = std::shared_ptr<abcdk_object_t>(NULL);

                if (envs && envs[0])
                    obj_envs = std::shared_ptr<abcdk_object_t>(abcdk_strtok2vector(envs, "\n"), [](void *p)
                                                               {if(p){abcdk_object_unref((abcdk_object_t**)&p);} });

                std::string cmdline;

                if (u_id != UINT32_MAX || g_id != UINT32_MAX)
                    cmdline = string_format("pkexec --user root %s", cmd);
                else
                    cmdline = string_format("%s", cmd);

                return abcdk_popen(cmdline.c_str(), (obj_envs.get() ? obj_envs->pstrs : NULL), u_id, g_id, rpath, wpath, stdin_fd, stdout_fd, stderr_fd);
            }

        } // namespace UtilEx

    } // namespace common
} // namespace abcdk

#endif // ABCDK_COMMON_UTILEX_HXX