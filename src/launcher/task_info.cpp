/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_info.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        std::shared_ptr<task_info> task_info::newTask(const uint64_t uuid)
        {
            return newTask(std::to_string(uuid));
        }

        std::shared_ptr<task_info> task_info::newTask(const std::string &uuid)
        {
            std::shared_ptr<task_info> one = std::shared_ptr<task_info>(new task_info(uuid.c_str()), [](void *p)
                                                                        {if(p){delete (task_info*)p;} });

            return one;
        }

        const char *task_info::getAppName()
        {
            if (m_name.empty())
                return ABCDK_GETTEXT("未命名");

            return m_name.c_str();
        }

        QIcon task_info::getAppIcon()
        {
            if (m_logo.empty())
                return QIcon("");

            QPixmap logo = QPixmap(m_logo.c_str());
            if (logo.isNull())
                return QIcon("");

            return QIcon(logo.scaled(QSize(256, 256), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }

        const char *task_info::uuid()
        {
            return m_uuid.c_str();
        }

        bool task_info::isAlive()
        {
            return false;
        }

        bool task_info::fetchLog(std::vector<char> &out, std::vector<char> &err)
        {
            ABCDK_UNUSED(out);
            ABCDK_UNUSED(err);

            return false;
        }

        int task_info::start()
        {
            std::string cmdline;

            uid_t uid = (m_uid.empty() ? 0 : atoi(m_uid.c_str()));
            gid_t gid = (m_gid.empty() ? 0 : atoi(m_gid.c_str()));

            if (uid != 0 || gid != 0)
                cmdline = common::UtilEx::string_format("pkexec --user root %s", m_exec.c_str());
            else
                cmdline = common::UtilEx::string_format("%s", m_exec.c_str());

            std::shared_ptr<abcdk_object_t> envs ;

            if (!m_env.empty())
                envs = std::shared_ptr<abcdk_object_t>(abcdk_strtok2vector(m_env.c_str(), "\n"), [](void *p)
                                                       {if(p){abcdk_object_unref((abcdk_object_t**)&p);} });
            else
                envs = std::shared_ptr<abcdk_object_t>(NULL);

            m_pid_fd = abcdk_popen(cmdline.c_str(), (envs.get()?envs->pstrs:NULL), uid, gid, (m_rwd.empty() ? NULL : m_rwd.c_str()), (m_cwd.empty() ? NULL : m_rwd.c_str()), NULL, &m_out_fd, &m_err_fd);
            if (m_pid_fd < 0)
                return -127;

            // 设置为非阻塞.
            abcdk_fflag_add(m_out_fd, O_NONBLOCK);
            abcdk_fflag_add(m_err_fd, O_NONBLOCK);

            return 0;
        }

        int task_info::stop()
        {
            return 0;
        }

        void task_info::deInit()
        {
        }

        void task_info::Init(const std::string &uuid)
        {
            m_tab_index = -1;

            m_uuid = uuid;
            m_create_usec = abcdk_time_realtime(3); // millisecond.
            m_pid_fd = -1;
            m_out_fd = -1;
            m_err_fd = -1;
            m_killed_cnt = 0;
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
