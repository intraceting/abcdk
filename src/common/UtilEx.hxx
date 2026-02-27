/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_UTILEX_HXX
#define ABCDK_COMMON_UTILEX_HXX

#include "abcdk.h"
#include "json/json.h"

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
                pid_t cid = -1;

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

                cid = abcdk_popen(cmdline.c_str(), (obj_envs.get() ? obj_envs->pstrs : NULL), u_id, g_id, rpath, wpath, stdin_fd, stdout_fd, stderr_fd);
                if (cid < 0)
                {
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("执行外部命令(%s)失败(errno=%d).\n"), errno);
                    return -1;
                }

                /*使子进程成为进程组的组长, 以便后续可以通过进程组管理.*/
                setpgid(cid, cid);

                abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("执行外部命令(%s)成功, 新的子进程(PID=%d)已经托管到后台执行.\n"), cmdline.c_str(), cid);

                return cid;
            }

#if defined(HAVE_SQLITE)
            static inline int sqlite_check_table_exist(sqlite3 *db, const char *name)
            {
                std::string sql = common::UtilEx::string_format("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='%s';", name);
                int chk;

                auto stmt = abcdk_sqlite_prepare(db, sql.c_str());
                if (!stmt)
                    return -1;

                chk = abcdk_sqlite_step(stmt);
                if (chk <= 0)
                {
                    abcdk_sqlite_finalize(stmt);
                    return -2;
                }

                chk = sqlite3_column_int(stmt, 0);
                abcdk_sqlite_finalize(stmt);

                return chk;
            }
#endif //#if defined(HAVE_SQLITE)
        
#if defined(HAVE_JSONCPP)
            static inline int jsoncpp_reader_parse_memory(const char *str, Json::Value &doc)
            {
                Json::Reader reader;

                bool bchk = reader.parse(str, doc);
                if(!bchk)
                    return -1;

                return 0;
            }

            static inline int jsoncpp_reader_parse_file(const char *file, Json::Value &doc)
            {
                int chk;

                abcdk_object_t *dump_data = abcdk_object_copyfrom_file(file);
                if (!dump_data)
                    return -1;

                chk = jsoncpp_reader_parse_memory(dump_data->pstrs[0], doc);
                abcdk_object_unref(&dump_data);

                return chk;
            }

            static inline std::string jsoncpp_writer_to_string(const Json::Value &doc)
            {
                return Json::FastWriter().write(doc);
            }

            static inline int jsoncpp_writer_to_file(const char *file, const Json::Value &doc)
            {
                std::string dump_data = jsoncpp_writer_to_string(doc);

                ssize_t wr_size = abcdk_dump(file, dump_data.data(), dump_data.size());
                if (wr_size <= 0 || (size_t)wr_size != dump_data.size())
                    return -1;

                return 0;
            }
#endif //#if defined(HAVE_JSONCPP)
        } // namespace UtilEx

    } // namespace common
} // namespace abcdk

#endif // ABCDK_COMMON_UTILEX_HXX