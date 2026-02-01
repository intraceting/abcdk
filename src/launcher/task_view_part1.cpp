/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_view_part1.hxx"


#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        void task_view_part1::deInit()
        {

        }

        void task_view_part1::Init()
        {
            QHBoxLayout *layout = new QHBoxLayout(this);
            layout->setContentsMargins(0,0,0,0);
            layout->setSpacing(8);

            m_edit_cmd = new common::QLineEditEx(this);
            m_edit_cmd->setPlaceholderText("在这里输入命令或点击右侧的配置按钮.");

            
            m_btn_conf = new common::QPushButtonEx(this);
            m_btn_conf->setIcon(QIcon(":/images/set.svg"));
            m_btn_conf->setToolTip("配置");

            layout->addWidget(m_edit_cmd,99);
            layout->addWidget(m_btn_conf,1);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
