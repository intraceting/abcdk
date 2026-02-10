/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_info.hxx"

#ifdef HAVE_QT5

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

        bool task_info::alive()
        {
            return !abcdk_atomic_compare(&m_child_state, 0);
        }

        int task_info::start()
        {
            if (!abcdk_atomic_compare(&m_child_state, 0))
                return 0;

            // 回收旧的线程资源.
            if (m_child_thread.joinable())
                m_child_thread.join();

            // 启动线程去异步处理.
            abcdk_atomic_store(&m_child_state, 0);
            m_child_thread = std::thread(&task_info::childRun, this);

            return 0;
        }

        int task_info::stop()
        {
            if (abcdk_atomic_compare_and_swap(&m_child_state, 1, 2))
                return 0;
            else if (abcdk_atomic_compare_and_swap(&m_child_state, 2, 3))
                return 0;
            else if (abcdk_atomic_compare_and_swap(&m_child_state, 3, 4))
                return 0;

            return -1;
        }

        ssize_t task_info::fetch(std::vector<char> &msg, bool out_or_err)
        {
            ssize_t chk_size;

            msg.resize(200 * 1024);

            if (out_or_err)
                chk_size = abcdk_stream_read(m_out_buf.get(), msg.data(), msg.size());
            else
                chk_size = abcdk_stream_read(m_err_buf.get(), msg.data(), msg.size());

            return chk_size;
        }

        void task_info::childRun()
        {
            pid_t exec_pid = -1;
            int exec_out_fd = -1;
            int exec_err_fd = -1;
            int exec_exitcode = 0;
            int exec_sigcode = 0;
            pid_t killer_pid = -1;
            int killer_exitcode = 0;
            int killer_sigcode = 0;
            pid_t chk_pid;

            std::thread m_stdout_thread;
            std::thread m_stderr_thread;

            while (1)
            {
                if (abcdk_atomic_compare_and_swap(&m_child_state, 0, 1))
                {
                    exec_pid = common::UtilEx::popen(m_uid.c_str(), m_gid.c_str(), m_env.c_str(), m_rwd.c_str(), m_cwd.c_str(), m_exec.c_str(), NULL, &exec_out_fd, &exec_err_fd);
                    if (exec_pid > 0)
                    {
                        m_stdout_thread = std::thread(&task_info::childStdout, this, exec_out_fd);
                        m_stderr_thread = std::thread(&task_info::childStderr, this, exec_err_fd);

                        abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("作业(%s)(%s)进程(PID=%d)已启动."), m_name.c_str(), m_exec.c_str(), exec_pid);
                    }
                    else
                    {
                        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("作业(%s)(%s)进程启动失败."), m_name.c_str(), m_exec.c_str());
                        break;
                    }
                }
                else if (abcdk_atomic_compare(&m_child_state, 2))
                {
                    if (!m_kill.empty())
                    {
                        killer_pid = common::UtilEx::popen(m_uid.c_str(), m_gid.c_str(), m_env.c_str(), m_rwd.c_str(), m_cwd.c_str(), m_kill.c_str(), NULL, NULL, NULL);
                        if (killer_pid > 0)
                        {
                            abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("杀手(%s)(%s)进程已经启动(PID=%d)."), m_name.c_str(), m_kill.c_str(), killer_pid);
                        }
                        else
                        {
                            abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("杀手(%s)(%s)进程启动失败."), m_name.c_str(), m_kill.c_str());
                        }
                    }
                    else
                    {
                        kill(exec_pid, SIGTERM);
                    }
                }
                else if (abcdk_atomic_compare(&m_child_state, 3))
                {
                    kill(exec_pid, SIGTERM);
                }
                else if (abcdk_atomic_compare(&m_child_state, 4))
                {
                    kill(exec_pid, SIGKILL);
                }

                pid_t chk_pid = abcdk_waitpid(exec_pid, WNOHANG, &exec_exitcode, &exec_sigcode);
                if (chk_pid > 0)
                {
                    abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("作业(%s)(%s)进程(PID=%d)已结束(EXITCODE=%d,SIGCODE=%d)."), m_name.c_str(), m_exec.c_str(), chk_pid, exec_exitcode, exec_sigcode);

                    // 如果杀手进程没有启动则直接退出.
                    if (killer_pid < 0)
                        break;

                    chk_pid = abcdk_waitpid(killer_pid, WNOHANG, &killer_exitcode, &killer_sigcode);
                    if (chk_pid > 0)
                    {
                        abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("杀手(%s)(%s)进程(PID=%d)已结束(EXITCODE=%d,SIGCODE=%d)."), m_name.c_str(), m_kill.c_str(), chk_pid, killer_exitcode, killer_sigcode);
                        break;
                    }
                    else
                    {
                        // 强制杀掉.
                        kill(killer_pid, SIGKILL);
                        abcdk_waitpid(killer_pid, WNOHANG, NULL, NULL);
                    }
                }

                usleep(300 * 1000); // 300 milliseconds.
            }

            if (m_stdout_thread.joinable())
                m_stdout_thread.join();
            if (m_stderr_thread.joinable())
                m_stderr_thread.join();

            abcdk_atomic_store(&m_child_state, 0);
        }

        void task_info::childStdout(int stdout_fd)
        {
            std::vector<char> buf(10 * 1024);
            int chk_size;

            while (1)
            {
                chk_size = read(stdout_fd, buf.data(), buf.size());
                if (chk_size <= 0)
                    break;

                abcdk_stream_write_buffer(m_out_buf.get(), buf.data(), chk_size);
            }

            abcdk_closep(&stdout_fd);
        }

        void task_info::childStderr(int stderr_fd)
        {
            std::vector<char> buf(10 * 1024);
            int chk_size;

            while (1)
            {
                chk_size = read(stderr_fd, buf.data(), buf.size());
                if (chk_size <= 0)
                    break;

                abcdk_stream_write_buffer(m_err_buf.get(), buf.data(), chk_size);
            }

            abcdk_closep(&stderr_fd);
        }

        void task_info::deInit()
        {
            while (!abcdk_atomic_compare(&m_child_state, 0))
            {
                stop();
                usleep(300 * 1000); // 300 milliseconds.
            }

            // 等待线程结束.
            if (m_child_thread.joinable())
                m_child_thread.join();
        }

        void task_info::Init(const std::string &uuid)
        {
            m_tab_index = -1;

            m_uuid = uuid;
            m_create_usec = abcdk_time_realtime(3); // millisecond.
            m_out_buf = std::shared_ptr<abcdk_stream_t>(abcdk_stream_create(), [](void *p)
                                                        {if(p){abcdk_stream_destroy((abcdk_stream_t**)&p);} });
            m_err_buf = std::shared_ptr<abcdk_stream_t>(abcdk_stream_create(), [](void *p)
                                                        {if(p){abcdk_stream_destroy((abcdk_stream_t**)&p);} });

            m_child_state = 0;
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
