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
#include "../common/Qt.hxx"
#include "../common/UtilEx.hxx"

#ifdef HAVE_QT

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
            pid_t m_pid_fd;
            int m_out_fd;
            int m_err_fd;
            int m_killed_cnt;

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
            bool isAlive();
            bool fetchLog(std::vector<char> &out, std::vector<char> &err);
            int start();
            int stop();
        protected:
            void deInit();
            void Init(const std::string &uuid);
        };
    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_LAUNCHER_TASK_CONFIG_HXX
