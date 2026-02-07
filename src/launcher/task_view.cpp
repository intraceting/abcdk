/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_view.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        std::shared_ptr<task_info> task_view::getInfo()
        {
            return m_info;
        }

        void task_view::onPopConfig()
        {
            int chk;
            task_config pop(m_info, this);
            pop.resize(600, 300);

            chk = pop.exec();

            abcdk_trace_printf(LOG_DEBUG, "%s, chk = %d", __FUNCTION__,  chk);

            if (chk != 1)
                return;

            m_edit_exec->setText(m_info->m_exec.c_str());
            emit updateState(m_info);//广播通知.

        }

        void task_view::onStart()
        {
            m_info->start();
        }

        void task_view::onStop()
        {
        }

        void task_view::deInit()
        {
        }

        void task_view::Init(std::shared_ptr<task_info> &info)
        {
            m_info = info;

            QVBoxLayout *layout = new QVBoxLayout(this);
            layout->setContentsMargins(10, 10, 10, 10);
            layout->setSpacing(8);

            QHBoxLayout *layout_part1 = new QHBoxLayout(NULL);
            layout_part1->setContentsMargins(0, 0, 0, 0);
            layout_part1->setSpacing(8);

            m_edit_exec = new common::QLineEditEx(this);
            m_edit_exec->setPlaceholderText(ABCDK_GETTEXT("在这里输入命令或点击右侧的配置按钮."));

            m_btn_conf = new common::QPushButtonEx(this);
            m_btn_conf->setIcon(QIcon(":/images/set.svg"));
            m_btn_conf->setToolTip(ABCDK_GETTEXT("配置"));

            layout_part1->addWidget(m_edit_exec, 99);
            layout_part1->addWidget(m_btn_conf, 1);

            QHBoxLayout *layout_part2 = new QHBoxLayout(NULL);
            layout_part2->setContentsMargins(0, 0, 0, 0);
            layout_part2->setSpacing(8);

            m_btn_clear = new common::QPushButtonEx(this);
            m_btn_clear->setText(ABCDK_GETTEXT("(&C)清空日志视图"));

            m_chk_autoroll = new common::QCheckBoxEx(this);
            m_chk_autoroll->setText(ABCDK_GETTEXT("(&F)显示最新日志"));

            m_btn_start = new common::QPushButtonEx(this);
            m_btn_start->setText(ABCDK_GETTEXT("(&R)启动"));

            m_btn_stop = new common::QPushButtonEx(this);
            m_btn_stop->setText(ABCDK_GETTEXT("(&K)停止"));

            connect(m_btn_start,&common::QPushButtonEx::clicked,this,&task_view::onStart);
            connect(m_btn_stop,&common::QPushButtonEx::clicked,this,&task_view::onStop);

            layout_part2->addWidget(m_btn_clear, 1);
            layout_part2->addWidget(m_chk_autoroll, 1);
            layout_part2->addStretch(96);
            layout_part2->addWidget(m_btn_start, 1);
            layout_part2->addWidget(m_btn_stop, 1);

            common::QTabWidgetEx *layout_part3 = new common::QTabWidgetEx(this);

            m_edit_stdout = new common::QPlainTextEditEx(this);
            m_edit_stdout->setReadOnly(true);
            m_edit_stdout->setMaximumBlockCount(10000);
            m_edit_stdout->setPlaceholderText(ABCDK_GETTEXT("这里将显示输出管道日志, 最新的日志在视图底部."));

            m_edit_stderr = new common::QPlainTextEditEx(this);
            m_edit_stderr->setReadOnly(true);
            m_edit_stderr->setMaximumBlockCount(10000);
            m_edit_stderr->setPlaceholderText(ABCDK_GETTEXT("这里将显示错误管道日志, 最新的日志在视图底部."));



            layout_part3->addTab(m_edit_stdout, ABCDK_GETTEXT("stdout"));
            layout_part3->addTab(m_edit_stderr, ABCDK_GETTEXT("stderr"));

            layout->addLayout(layout_part1, 1);
            layout->addWidget(layout_part3, 98);
            layout->addLayout(layout_part2, 2);

            connect(m_btn_conf, &QPushButton::clicked, this, &task_view::onPopConfig);
        }

        void task_view::mousePressEvent(QMouseEvent *event)
        {
            if (event->button() == Qt::RightButton)
            {
                event->accept(); // 拦截鼠标右键单击事件,以防止传递给父窗体.
                return;
            }

            common::QWidgetEx::mousePressEvent(event);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
