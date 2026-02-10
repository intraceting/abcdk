/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_METADATA_HXX
#define ABCDK_LAUNCHER_METADATA_HXX

#include <iostream>
#include <memory>
#include <mutex>
#include "abcdk.h"
#include "QObjectEx.hxx"
#include "task_info.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        class metadata : public common::QObjectEx
        {
            Q_OBJECT
        public:
            int m_help_show;
            /**PID文件.*/
            std::vector<char> m_pid_file_default;
            std::string m_pid_file;
            /**日志文件.*/
            std::vector<char> m_log_file_default;
            std::string m_log_file;
            /**详细日志关开.*/
            int m_log_verbose;
            /*语言和编码.*/
            std::vector<char> m_lang_codeset_default;
            std::string m_lang_codeset;
            /*语言文件域名.*/
            std::vector<char> m_lang_file_name_default;
            std::string m_lang_file_name;
            /*语言文件路径.*/
            std::vector<char> m_lang_file_path_default;
            std::string m_lang_file_path;
            /*用户家目录.*/
            std::vector<char> m_user_home_path;
            /*缓存目录.*/
            std::vector<char> m_cache_path_default;
            std::string m_cache_path;

            size_t m_alive_tasks_count;

            std::mutex m_tasks_mutex;
            std::map<std::string, std::shared_ptr<task_info>> m_tasks;

        private:
            abcdk_option_t *m_args;

        protected:
            metadata(QObject *parent = nullptr)
                : common::QObjectEx(parent)
            {
                Init();
            }

            virtual ~metadata()
            {
                deInit();
            }

        public:
            static std::shared_ptr<metadata> get();

        public:
            void parseCmdLine(int &argc, char *argv[]);
            void printUsage(FILE *out = stderr);
            bool isPrintUsage();

        protected:
            void deInit();
            void Init();
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_LAUNCHER_METADATA_HXX
