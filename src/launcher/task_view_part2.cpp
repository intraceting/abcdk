/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_view_part2.hxx"


#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        void task_view_part2::deInit()
        {

        }

        void task_view_part2::Init()
        {
            QHBoxLayout *layout = new QHBoxLayout(this);
            layout->setContentsMargins(0,0,0,0);
            layout->setSpacing(8);

            m_btn_clear = new common::QPushButtonEx(this);
            m_btn_clear->setText("(&C)清空日志视图");

            m_chk_autoroll = new common::QCheckBoxEx(this);
            m_chk_autoroll->setText("(&F)显示最新日志");

            m_btn_start = new common::QPushButtonEx(this);
            m_btn_start->setText("(&R)启动");

            m_btn_stop = new common::QPushButtonEx(this);
            m_btn_stop->setText("(&K)停止");

            layout->addWidget(m_btn_clear,1);
            layout->addWidget(m_chk_autoroll,1);
            layout->addStretch(96);
            layout->addWidget(m_btn_start,1);
            layout->addWidget(m_btn_stop,1);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
