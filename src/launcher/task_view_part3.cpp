/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_view_part3.hxx"


#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        void task_view_part3::deInit()
        {

        }

        void task_view_part3::Init()
        {
            m_edit_stdout = new common::QPlainTextEditEx(this);
            m_edit_stdout->setReadOnly(true);
            m_edit_stdout->setMaximumBlockCount(10000);
            m_edit_stdout->setPlaceholderText("这里将显示输出管道日志, 最新的日志在视图底部.");

            m_edit_stderr = new common::QPlainTextEditEx(this);
            m_edit_stderr->setReadOnly(true);
            m_edit_stderr->setMaximumBlockCount(10000);
            m_edit_stderr->setPlaceholderText("这里将显示错误管道日志, 最新的日志在视图底部.");

            addTab(m_edit_stdout, "stdout");
            addTab(m_edit_stderr, "stderr");
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
