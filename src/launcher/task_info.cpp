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

            return common::QUtilEx::getIcon(m_logo.c_str());
        }

        const char *task_info::uuid()
        {
            return m_uuid.c_str();
        }

        bool task_info::alive()
        {
            int exec_exitcode = 0;
            int exec_sigcode = 0;
            int killer_exitcode = 0;
            int killer_sigcode = 0;
            pid_t chk_pid;

            if (m_exec_pid > 0)
            {
                chk_pid = abcdk_waitpid(m_exec_pid, WNOHANG, &exec_exitcode, &exec_sigcode);
                if (chk_pid > 0)
                {
                    m_exec_pid = -1;
                    abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("作业(%s)(%s)进程(PID=%d)已结束(EXITCODE=%d,SIGCODE=%d)."), m_name.c_str(), m_exec.c_str(), chk_pid, exec_exitcode, exec_sigcode);
                }
            }

            if (m_killer_pid > 0)
            {
                chk_pid = abcdk_waitpid(m_killer_pid, WNOHANG, &killer_exitcode, &killer_sigcode);
                if (chk_pid > 0)
                {
                    m_killer_pid = -1;
                    abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("杀手(%s)(%s)进程(PID=%d)已结束(EXITCODE=%d,SIGCODE=%d)."), m_name.c_str(), m_killer_exec.c_str(), chk_pid, killer_exitcode, killer_sigcode);
                }
            }

            if (m_exec_pid > 0 || m_killer_pid > 0)
                return true;

            abcdk_atomic_store(&m_child_state, 0);
            return false;
        }

        int task_info::start()
        {
            if (abcdk_atomic_compare_and_swap(&m_child_state, 0, 1))
            {
                m_exec_pid = -1;
                m_exec_out_fd = -1;
                m_exec_err_fd = -1;
                m_killer_exec = "";
                m_killer_pid = -1;

                if (m_exec.empty())
                {
                    abcdk_atomic_store(&m_child_state, 0);
                    return -1;
                }

                m_exec_pid = common::UtilEx::popen(m_uid.c_str(), m_gid.c_str(), m_env.c_str(), m_rwd.c_str(), m_cwd.c_str(), m_exec.c_str(), NULL, &m_exec_out_fd, &m_exec_err_fd);
                if (m_exec_pid < 0)
                {
                    abcdk_atomic_store(&m_child_state, 0);
                    return -1;
                }

                // 回收旧的线程资源.
                if (m_stdout_thread.joinable())
                    m_stdout_thread.join();
                if (m_stderr_thread.joinable())
                    m_stderr_thread.join();

                // 启动线程去异步处理.
                m_stdout_thread = std::thread(&task_info::childStdout, this, m_exec_out_fd);
                m_stderr_thread = std::thread(&task_info::childStderr, this, m_exec_err_fd);
            }

            return 0;
        }

        int task_info::stop()
        {
            int killer_exitcode = 0;
            int killer_sigcode = 0;
            pid_t chk_pid;

            if (abcdk_atomic_compare_and_swap(&m_child_state, 1, 2))
            {
                if (!m_kill.empty())
                {
                    m_killer_exec = m_kill;
                    m_killer_pid = common::UtilEx::popen(m_uid.c_str(), m_gid.c_str(), m_env.c_str(), m_rwd.c_str(), m_cwd.c_str(), m_killer_exec.c_str(), NULL, NULL, NULL);
                }
                else
                {
                    m_killer_exec = common::UtilEx::string_format("kill -s %d -%d", SIGTERM, getpgid(m_exec_pid));
                    m_killer_pid = common::UtilEx::popen(m_uid.c_str(), m_gid.c_str(), NULL, NULL, NULL, m_killer_exec.c_str(), NULL, NULL, NULL);
                }

                return (m_killer_pid > 0 ? 0 : -1);
            }
            else if (abcdk_atomic_compare_and_swap(&m_child_state, 2, 3))
            {
                if (m_killer_pid > 0)
                {
                    chk_pid = abcdk_waitpid(m_killer_pid, WNOHANG, &killer_exitcode, &killer_sigcode);
                    if (chk_pid <= 0)
                    {
                        abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("杀手(%s)(%s)进程(PID=%d)正在运行..."), m_name.c_str(), m_killer_exec.c_str(), m_killer_pid);
                        return -EAGAIN;
                    }
                    else
                    {
                        m_killer_pid = -1;
                        abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("杀手(%s)(%s)进程(PID=%d)已结束(EXITCODE=%d,SIGCODE=%d)."), m_name.c_str(), m_killer_exec.c_str(), chk_pid, killer_exitcode, killer_sigcode);
                    }
                }

                m_killer_exec = common::UtilEx::string_format("kill -s %d -%d", SIGTERM, getpgid(m_exec_pid));
                m_killer_pid = common::UtilEx::popen(m_uid.c_str(), m_gid.c_str(), NULL, NULL, NULL, m_killer_exec.c_str(), NULL, NULL, NULL);

                return (m_killer_pid > 0 ? 0 : -1);
            }
            else if (abcdk_atomic_compare_and_swap(&m_child_state, 3, 4))
            {
                if (m_killer_pid > 0)
                {
                    chk_pid = abcdk_waitpid(m_killer_pid, WNOHANG, &killer_exitcode, &killer_sigcode);
                    if (chk_pid <= 0)
                    {
                        abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("杀手(%s)(%s)进程(PID=%d)正在运行..."), m_name.c_str(), m_killer_exec.c_str(), m_killer_pid);
                        return -EAGAIN;
                    }
                    else
                    {
                        m_killer_pid = -1;
                        abcdk_trace_printf(LOG_INFO, ABCDK_GETTEXT("杀手(%s)(%s)进程(PID=%d)已结束(EXITCODE=%d,SIGCODE=%d)."), m_name.c_str(), m_killer_exec.c_str(), chk_pid, killer_exitcode, killer_sigcode);
                    }
                }

                m_killer_exec = common::UtilEx::string_format("kill -s %d -%d", SIGKILL, getpgid(m_exec_pid));
                m_killer_pid = common::UtilEx::popen(m_uid.c_str(), m_gid.c_str(), NULL, NULL, NULL, m_killer_exec.c_str(), NULL, NULL, NULL);

                return (m_killer_pid > 0 ? 0 : -1);
            }

            return -1;
        }

        ssize_t task_info::fetch(std::vector<char> &msg, bool out_or_err)
        {
            ssize_t chk_size;

            msg.clear();
            msg.resize(200 * 1024);

            if (out_or_err)
                chk_size = abcdk_stream_read(m_out_buf.get(), msg.data(), msg.size());
            else
                chk_size = abcdk_stream_read(m_err_buf.get(), msg.data(), msg.size());

            return chk_size;
        }

        void task_info::childStdout(int stdout_fd)
        {
            std::vector<char> buf(10 * 1024);
            int chk_size;

            if (stdout_fd < 0)
                return;

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

            if (stderr_fd < 0)
                return;

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
            while (alive())
            {
                stop();
                usleep(300 * 1000); // 300 milliseconds.
            }

            // 回收旧的线程资源.
            if (m_stdout_thread.joinable())
                m_stdout_thread.join();
            if (m_stderr_thread.joinable())
                m_stderr_thread.join();
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

            m_exec_pid = -1;
            m_exec_out_fd = -1;
            m_exec_err_fd = -1;
            m_killer_exec = "";
            m_killer_pid = -1;
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
