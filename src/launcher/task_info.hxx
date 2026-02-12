/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_TASK_INFO_HXX
#define ABCDK_LAUNCHER_TASK_INFO_HXX

#include "abcdk.h"
#include <iostream>
#include <memory>
#include <thread>
#include "../common/QUtilEx.hxx"
#include "../common/UtilEx.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        class task_info
        {
        public:
            int m_tab_index;
            std::string m_name;
            std::string m_logo;
            std::string m_exec;
            std::string m_kill;
            std::string m_rwd;
            std::string m_cwd;
            std::string m_uid;
            std::string m_gid;
            std::string m_env;
        private:
            std::string m_uuid;
            uint64_t m_create_usec;
            std::shared_ptr<abcdk_stream_t> m_out_buf;
            std::shared_ptr<abcdk_stream_t> m_err_buf;
            int m_child_state;
            std::thread m_child_thread;

        public:
            task_info(const std::string &uuid = "")
            {
                Init(uuid);
            }

            virtual ~task_info()
            {
                deInit();
            }
        public:
            static std::shared_ptr<task_info> newTask(const uint64_t uuid);
            static std::shared_ptr<task_info> newTask(const std::string &uuid);
        public:
            const char *getAppName();
            QIcon getAppIcon();
        public:
            const char *uuid();
            bool alive();
            int start();
            int stop();
            ssize_t fetch(std::vector<char> &msg, bool out_or_err);
        protected:
            void childRun();
            void childStdout(int stdout_fd);
            void childStderr(int stderr_fd);
            void deInit();
            void Init(const std::string &uuid);
        };
    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_LAUNCHER_TASK_CONFIG_HXX
