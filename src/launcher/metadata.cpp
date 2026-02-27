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

        void metadata::loadTasks()
        {
            int chk;
            Json::Value doc;

            std::string dump_file = m_cache_path + "/" + m_main_db_name;

            chk = common::UtilEx::jsoncpp_reader_parse_file(dump_file.c_str(),doc);
            if(chk != 0)
            {
                abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("读缓存文件(%s)失败, 不存在或无权限."), dump_file.c_str());
                return;
            }

            try
            {
                int ver_major = doc["abcdk"]["launcher"]["version"]["major"].asInt();
                int ver_minor = doc["abcdk"]["launcher"]["version"]["minor"].asInt();
                int ver_patch = doc["abcdk"]["launcher"]["version"]["patch"].asInt();

                Json::Value tasks = doc["abcdk"]["launcher"]["tasks"];
                for (Json::Value::ArrayIndex i = 0; i < tasks.size(); ++i)
                {
                    auto &one_task = tasks[i];

                    std::string uuid = one_task["uuid"].asCString();
                    if(uuid.empty())
                        continue;

                    std::shared_ptr<abcdk::launcher::task_info> one_info = task_info::newTask(uuid);

                    one_info->m_index = one_task["index"].asInt();
                    one_info->m_uuid = uuid;
                    one_info->m_name = one_task["name"].asCString();
                    one_info->m_logo = one_task["logo"].asCString();
                    one_info->m_exec = one_task["exec"].asCString();
                    one_info->m_kill = one_task["kill"].asCString();
                    one_info->m_rwd = one_task["rwd"].asCString();
                    one_info->m_cwd = one_task["cwd"].asCString();
                    one_info->m_uid = one_task["uid"].asCString();
                    one_info->m_gid = one_task["gid"].asCString();
                    one_info->m_env = one_task["env"].asCString();

                    m_tasks[uuid] = one_info;
                }
            }
            catch(const std::exception& e)
            {
                abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("解析缓存数据失败, 原因是: '%s'"), e.what());
                return;
            }
            
        }

        void metadata::saveTasks()
        {
            int chk;

            Json::Value doc;

            doc["abcdk"]["launcher"]["version"]["major"] = ABCDK_VERSION_MAJOR;
            doc["abcdk"]["launcher"]["version"]["minor"] = ABCDK_VERSION_MINOR;
            doc["abcdk"]["launcher"]["version"]["patch"] = ABCDK_VERSION_PATCH;

            for (auto &one : m_tasks)
            {
                Json::Value one_task;

                one_task["index"] = one.second->m_index;
                one_task["uuid"] = one.second->m_uuid.c_str();
                one_task["name"] = one.second->m_name.c_str();
                one_task["logo"] = one.second->m_logo.c_str();
                one_task["exec"] = one.second->m_exec.c_str();
                one_task["kill"] = one.second->m_kill.c_str();
                one_task["rwd"] = one.second->m_rwd.c_str();
                one_task["cwd"] = one.second->m_cwd.c_str();
                one_task["uid"] = one.second->m_uid.c_str();
                one_task["gid"] = one.second->m_gid.c_str();
                one_task["env"] = one.second->m_env.c_str();

                doc["abcdk"]["launcher"]["tasks"].append(one_task);
            }

            std::string dump_file = m_cache_path + "/" + m_main_db_name;

            chk = common::UtilEx::jsoncpp_writer_to_file(dump_file.c_str(),doc);
            if (chk != 0)
            {
                abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("写缓存文件(%s)失败, 无空间或无权限."), dump_file.c_str());
                return;
            }
        }

        void metadata::deInit()
        {
            int chk;

            abcdk_option_free(&m_args);
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

            m_main_db_name = "main.xml";

            m_args = NULL;
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
