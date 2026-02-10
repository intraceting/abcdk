/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "metadata.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        std::shared_ptr<metadata> metadata::get()
        {
            static std::shared_ptr<metadata> only_one = std::shared_ptr<metadata>(new metadata, [](void *p)
                                                                                  {if(p){delete (metadata*)p;} });

            return only_one;
        }

        void metadata::parseCmdLine(int &argc, char *argv[])
        {
            m_args = abcdk_option_alloc("--");
            if (!m_args)
                return;

            abcdk_getargs(m_args, argc, argv);
            argc = 1; // 只保留第一个参数.

            m_help_show = abcdk_option_exist(m_args, "--help");

            m_pid_file = abcdk_option_get(m_args, "--pid-file", 0, m_pid_file_default.data());
            m_log_file = abcdk_option_get(m_args, "--log-file", 0, m_log_file_default.data());
            m_log_verbose = abcdk_option_get_int(m_args, "--log-verbose", 0, 0);

            m_lang_codeset = abcdk_option_get(m_args, "--lang-codeset", 0, m_lang_codeset_default.data());
            m_lang_file_name = abcdk_option_get(m_args, "--lang-file-name", 0, m_lang_file_name_default.data());
            m_lang_file_path = abcdk_option_get(m_args, "--lang-file-path", 0,  m_lang_file_path_default.data());

            
            m_cache_path = abcdk_option_get(m_args, "--cache-path", 0, m_cache_path_default.data());

        }

        void metadata::printUsage(FILE *out /*= stderr*/)
        {
            fprintf(out, "\n描述:\n");

            fprintf(out, "\n\t简单应用程序启动器.\n");

            fprintf(stderr, "\n\t--pid-file < FILE >\n");
            fprintf(stderr, ABCDK_GETTEXT("\t\tPID文件. 默认: %s\n"), m_pid_file_default.data());

            fprintf(stderr, "\n\t--log-file < FILE >\n");
            fprintf(stderr, ABCDK_GETTEXT("\t\t日志文件.默认: %s\n"), m_log_file_default.data());

            fprintf(stderr, "\n\t--log-verbose < BOOL >\n");
            fprintf(stderr, ABCDK_GETTEXT("\t\t记录详细日志. 默认: 0\n"));

            fprintf(stderr, ABCDK_GETTEXT("\n\t\t0: 否\n"));
            fprintf(stderr, ABCDK_GETTEXT("\t\t1: 是\n"));

            fprintf(stderr, "\n\t--lang-codeset < LANG.CODESET >\n");
            fprintf(stderr, ABCDK_GETTEXT("\t\t语言和编码. 默认: %s\n"), m_lang_codeset_default.data());

            fprintf(stderr, "\n\t--lang-file-name < NAME >\n");
            fprintf(stderr, ABCDK_GETTEXT("\t\t语言文件名称. 默认: %s\n"), m_lang_file_name_default.data());

            fprintf(stderr, "\n\t--lang-file-path < PATH >\n");
            fprintf(stderr, ABCDK_GETTEXT("\t\t语言文件路径. 默认: %s\n"), m_lang_file_path_default.data());

            fprintf(stderr, "\n\t--cache-path < PATH >\n");
            fprintf(stderr, ABCDK_GETTEXT("\t\t缓存路径. 默认: %s\n"), m_cache_path_default.data());
        }

        bool metadata::isPrintUsage()
        {
            return (m_help_show);
        }

        void metadata::deInit()
        {
            abcdk_option_free(&m_args);
        }

        void metadata::Init()
        {
            m_pid_file_default.resize(PATH_MAX);
            strcpy(m_pid_file_default.data(),"/tmp/abcdk/launcher/pid.lock");

            m_log_file_default.resize(PATH_MAX);
            strcpy(m_log_file_default.data(),"/tmp/abcdk/launcher/log.txt");

            m_lang_codeset_default.resize(NAME_MAX);
            strcpy(m_lang_codeset_default.data(),"zh_CN.UTF-8");

            m_lang_file_name_default.resize(NAME_MAX);
            strcpy(m_lang_file_name_default.data(),"abcdk-bin");

            m_lang_file_path_default.resize(PATH_MAX);
            strcpy(m_lang_file_path_default.data(),"../share/locale/");

            m_user_home_path.resize(PATH_MAX);
            abcdk_user_dir_home(m_user_home_path.data(), NULL);

            m_cache_path_default.resize(PATH_MAX);
            abcdk_user_dir_home(m_cache_path_default.data(), ".cache/abcdk/launcher/");

            m_alive_tasks_count = 0;

            m_args = NULL;
        }
    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
