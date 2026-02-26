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
            m_lang_file_path = abcdk_option_get(m_args, "--lang-file-path", 0, m_lang_file_path_default.data());

            m_cache_path = abcdk_option_get(m_args, "--cache-path", 0, m_cache_path_default.data());
        }

        void metadata::printUsage(FILE *out /*= stderr*/)
        {
            char name[NAME_MAX] = {0};

            abcdk_proc_basename(name);

            fprintf(out, "\n关于:\n");

            fprintf(out, "\n\t名称: 应用程序启动器\n");
            fprintf(out, "\n\t版本: %d.%d.%d\n", ABCDK_VERSION_MAJOR, ABCDK_VERSION_MINOR, ABCDK_VERSION_PATCH);

            fprintf(out, "\n选项:\n");

            fprintf(out, "\n\t--pid-file < FILE >\n");
            fprintf(out, ABCDK_GETTEXT("\t\tPID文件. 默认: %s\n"), m_pid_file_default.data());

            fprintf(out, "\n\t--log-file < FILE >\n");
            fprintf(out, ABCDK_GETTEXT("\t\t日志文件.默认: %s\n"), m_log_file_default.data());

            fprintf(out, "\n\t--log-verbose < BOOL >\n");
            fprintf(out, ABCDK_GETTEXT("\t\t记录详细日志. 默认: 0\n"));

            fprintf(out, ABCDK_GETTEXT("\n\t\t0: 否\n"));
            fprintf(out, ABCDK_GETTEXT("\t\t1: 是\n"));

            fprintf(out, "\n\t--lang-codeset < LANG.CODESET >\n");
            fprintf(out, ABCDK_GETTEXT("\t\t语言和编码. 默认: %s\n"), m_lang_codeset_default.data());

            fprintf(out, "\n\t--lang-file-name < NAME >\n");
            fprintf(out, ABCDK_GETTEXT("\t\t语言文件名称. 默认: %s\n"), m_lang_file_name_default.data());

            fprintf(out, "\n\t--lang-file-path < PATH >\n");
            fprintf(out, ABCDK_GETTEXT("\t\t语言文件路径. 默认: %s\n"), m_lang_file_path_default.data());

            fprintf(out, "\n\t--cache-path < PATH >\n");
            fprintf(out, ABCDK_GETTEXT("\t\t缓存路径. 默认: %s\n"), m_cache_path_default.data());
        }

        bool metadata::isPrintUsage()
        {
            return (m_help_show);
        }

        int metadata::checkCache()
        {
            static int chk_ok_once = 0;
            std::string sql;
            int chk, chk2, chk3, chk4;

            if (!abcdk_atomic_compare_and_swap(&chk_ok_once, 0, 1))
                return 0;

            m_main_db_pathfile = common::UtilEx::string_format("%s/%s", m_cache_path.c_str(), m_main_db_filename.c_str());

            if (access(m_main_db_pathfile.c_str(), F_OK) == 0)
            {
                m_main_db = abcdk_sqlite_open(m_main_db_pathfile.c_str());
                if (!m_main_db)
                {
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("打开缓存文件(%s)出错, 已损坏或无权限."), m_main_db_pathfile.c_str());
                    return -1;
                }

                abcdk_sqlite_journal_mode(m_main_db, ABCDK_SQLITE_JOURNAL_MEMORY);

                chk = common::UtilEx::sqlite_check_table_exist(m_main_db, ABCDK_GETTEXT("version"));
                if (chk <= 0)
                {
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("读缓存文件(%s)出错, 已损坏或不兼容."), m_main_db_pathfile.c_str());
                    chk = abcdk_sqlite_closep(&m_main_db);
                    ABCDK_TRACE_ASSERT(chk == SQLITE_OK, ABCDK_GETTEXT("关闭缓存文失败."));
                    return -2;
                }

                chk = common::UtilEx::sqlite_check_table_exist(m_main_db, "tasks");
                if (chk <= 0)
                {
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("读缓存文件(%s)出错, 已损坏或不兼容."), m_main_db_pathfile.c_str());
                    chk = abcdk_sqlite_closep(&m_main_db);
                    ABCDK_TRACE_ASSERT(chk == SQLITE_OK, ABCDK_GETTEXT("关闭缓存文失败."));
                    return -3;
                }
            }
            else
            {
                // 创建可能不存在的路径.
                abcdk_mkdir(m_main_db_pathfile.c_str(), 0755);

                m_main_db = abcdk_sqlite_open(m_main_db_pathfile.c_str());
                if (!m_main_db)
                {
                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("打开缓存文件(%s)出错, 无空间或无权限."), m_main_db_pathfile.c_str());
                    return -4;
                }

                abcdk_sqlite_journal_mode(m_main_db, ABCDK_SQLITE_JOURNAL_MEMORY);

                abcdk_sqlite_tran_begin(m_main_db);

                sql = "CREATE TABLE version ("
                      "major INTEGER DEFAULT (0) NOT NULL,"
                      "minor INTEGER DEFAULT (0) NOT NULL,"
                      "patch INTEGER DEFAULT (0) NOT NULL"
                      ");";

                chk2 = abcdk_sqlite_exec_direct(m_main_db, sql.c_str());

                sql = common::UtilEx::string_format("INSERT INTO version (major, minor, patch) VALUES(%d, %d, %d);",
                                                    ABCDK_VERSION_MAJOR, ABCDK_VERSION_MINOR, ABCDK_VERSION_PATCH);

                chk3 = abcdk_sqlite_exec_direct(m_main_db, sql.c_str());

                sql = "CREATE TABLE tasks ("
                      "task_index INTEGER DEFAULT (-1),"
                      "task_uuid TEXT(32) NOT NULL,"
                      "task_name TEXT(225),"
                      "task_logo TEXT(255),"
                      "task_exec TEXT(40960),"
                      "task_kill TEXT(4096),"
                      "task_uid TEXT(20),"
                      "task_gid TEXT(20),"
                      "task_rwd TEXT(4096),"
                      "task_cwd TEXT(4096),"
                      "task_env TEXT(40960),"
                      "CONSTRAINT tasks_pk PRIMARY KEY (task_uuid)"
                      ");";

                chk4 = abcdk_sqlite_exec_direct(m_main_db, sql.c_str());

                chk = ((chk2 >= 0 && chk3 >= 0 && chk4 >= 0) ? abcdk_sqlite_tran_commit(m_main_db) : SQLITE_ERROR);
                if (chk != SQLITE_OK)
                {
                    abcdk_sqlite_tran_rollback(m_main_db); // 回滚.

                    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("写缓存文件(%s)出错 ,无空间或无权限."), m_main_db_pathfile.c_str());
                    chk = abcdk_sqlite_closep(&m_main_db);
                    ABCDK_TRACE_ASSERT(chk == SQLITE_OK, ABCDK_GETTEXT("关闭缓存文失败."));
                    return -5;
                }
            }

            return 0;
        }

        void metadata::loadTasks()
        {
            int chk;

            chk = checkCache();
            if (chk != 0)
                return;

                
        }

        void metadata::saveTasks()
        {
            int chk;

            chk = checkCache();
            if (chk != 0)
                return;

            abcdk_sqlite_tran_begin(m_main_db);

            abcdk_sqlite_exec_direct(m_main_db, "DELETE FROM tasks;");

            for (auto &one : m_tasks)
            {
                std::vector<char> sql(200 * 1024);

                snprintf(sql.data(), sql.size(),
                         "INSERT INTO tasks (task_index, task_uuid, task_name,task_logo,"
                         "task_exec, task_kill, task_uid ,task_gid, task_rwd, task_cwd, task_env) "
                         "VALUES task_kill(%d, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
                         one.second->m_index, one.second->m_uuid.c_str(), one.second->m_logo.c_str(),
                         one.second->m_name.c_str(), one.second->m_exec.c_str(), one.second->m_kill.c_str(),
                         one.second->m_uid.c_str(), one.second->m_gid.c_str(),
                         one.second->m_rwd.c_str(), one.second->m_cwd.c_str(),
                         one.second->m_env.c_str());

                chk = abcdk_sqlite_exec_direct(m_main_db, sql.data());
                if(chk != SQLITE_OK)
                    break;
            }

            if(chk == SQLITE_OK)
                chk = abcdk_sqlite_tran_commit(m_main_db);

            if(chk != SQLITE_OK)
                abcdk_sqlite_tran_rollback(m_main_db);
            
        }

        void metadata::deInit()
        {
            int chk;

            abcdk_option_free(&m_args);

            chk = abcdk_sqlite_closep(&m_main_db);
            ABCDK_TRACE_ASSERT(chk == SQLITE_OK, "关闭缓存文失败.");
        }

        void metadata::Init()
        {
            m_pid_file_default.resize(PATH_MAX);
            strcpy(m_pid_file_default.data(), "/tmp/abcdk/launcher/pid.lock");

            m_log_file_default.resize(PATH_MAX);
            strcpy(m_log_file_default.data(), "/tmp/abcdk/launcher/log.txt");

            m_lang_codeset_default.resize(NAME_MAX);
            strcpy(m_lang_codeset_default.data(), "zh_CN.UTF-8");

            m_lang_file_name_default.resize(NAME_MAX);
            strcpy(m_lang_file_name_default.data(), "abcdk-bin");

            m_lang_file_path_default.resize(PATH_MAX);
            strcpy(m_lang_file_path_default.data(), "../share/locale/");

            m_user_home_path.resize(PATH_MAX);
            abcdk_user_dir_home(m_user_home_path.data(), NULL);

            m_cache_path_default.resize(PATH_MAX);
            abcdk_user_dir_home(m_cache_path_default.data(), ".cache/abcdk/launcher/");

            m_main_db_filename = "main.db";

            m_args = NULL;
            m_main_db = NULL;
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
