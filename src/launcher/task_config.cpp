/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_config.hxx"


#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        void task_config::onOpenIcon()
        {
            QString pathfile = QFileDialog::getOpenFileName(this,
                                                            ABCDK_GETTEXT("选择应用图标"),
                                                            metadata::get()->m_user_home_path.data(),
                                                            ABCDK_GETTEXT("ICON文件 (*.ico *.icon);;PNG文件 (*.png);;JPG文件 (*.jpg *.jpeg);;SVG文件 (*.svg);;所有文件 (*)"));

            if (pathfile.isEmpty())
                return;

            m_edit_logo->setText(pathfile);
        }

        void task_config::onOpenExec()
        {
            QString pathfile = QFileDialog::getOpenFileName(this,
                                                            ABCDK_GETTEXT("选择应用程序"),
                                                            metadata::get()->m_user_home_path.data(),
                                                            ABCDK_GETTEXT("所有文件 (*)"));

            if (pathfile.isEmpty())
                return;

            m_edit_exec->setText(pathfile);
        }

        void task_config::onOpenKill()
        {
            QString pathfile = QFileDialog::getOpenFileName(this,
                                                            ABCDK_GETTEXT("选择应用程序"),
                                                            metadata::get()->m_user_home_path.data(),
                                                            ABCDK_GETTEXT("所有文件 (*)"));

            if (pathfile.isEmpty())
                return;

            m_edit_kill->setText(pathfile);
        }

        void task_config::onCancle()
        {
            close();
        }

        void task_config::onSave()
        {
            m_info->m_name = m_edit_name->text().toStdString();
            m_info->m_logo = m_edit_logo->text().toStdString();
            m_info->m_exec = m_edit_exec->text().toStdString();
            m_info->m_kill = m_edit_kill->text().toStdString();
            m_info->m_rwd = m_edit_rwd->text().toStdString();
            m_info->m_cwd = m_edit_cwd->text().toStdString();
            m_info->m_uid = m_edit_uid->text().toStdString();
            m_info->m_gid = m_edit_gid->text().toStdString();
            m_info->m_env = m_edit_env->toPlainText().toStdString();

            if(m_info->m_name.empty())
            {
                QMessageBox::information(this, ABCDK_GETTEXT("提示"), ABCDK_GETTEXT("'名称'参数不能为空."));
                m_edit_name->setFocus();
                return;
            }

            if(m_info->m_exec.empty())
            {
                QMessageBox::information(this, ABCDK_GETTEXT("提示"), ABCDK_GETTEXT("'启动'参数不能为空."));
                m_edit_exec->setFocus();
                return;
            }

            done(1);
        }

        void task_config::deInit()
        {

        }

        void task_config::Init(std::shared_ptr<task_info> &info)
        {
            setObjectName("task_config");
            setWindowTitle(ABCDK_GETTEXT("配置"));

            m_info = info;
            
            QGridLayout *layout = new QGridLayout(this);
            layout->setContentsMargins(10,10,10,10);
            layout->setHorizontalSpacing(8);
            layout->setVerticalSpacing(8);

            m_lab_name = new common::QLabelEx(this);
            m_lab_name->setText(ABCDK_GETTEXT("名称:"));

            m_edit_name = new common::QLineEditEx(this);
            m_edit_name->setPlaceholderText(ABCDK_GETTEXT("在这里输入应用程序名称(仅用于显示)."));
            m_edit_name->setText(m_info->m_name.c_str());

            layout->addWidget(m_lab_name,0,0);
            layout->addWidget(m_edit_name,0,1);

            m_lab_logo = new common::QLabelEx(this);
            m_lab_logo->setText(ABCDK_GETTEXT("图标:"));

            m_edit_logo = new common::QLineEditEx(this);
            m_edit_logo->setPlaceholderText(ABCDK_GETTEXT("在这里输入应用程序图标文件(仅用于显示), 或双击打开选择对话框."));
            m_edit_logo->setText(m_info->m_logo.c_str());

            layout->addWidget(m_lab_logo,1,0);
            layout->addWidget(m_edit_logo,1,1);

            
            m_lab_exec = new common::QLabelEx(this);
            m_lab_exec->setText(ABCDK_GETTEXT("启动:"));

            m_edit_exec = new common::QLineEditEx(this);
            m_edit_exec->setPlaceholderText(ABCDK_GETTEXT("在这里输入应用程序启动(执行)文件, 或双击打开选择对话框."));
            m_edit_exec->setText(m_info->m_exec.c_str());
            
            layout->addWidget(m_lab_exec,2,0);
            layout->addWidget(m_edit_exec,2,1);

            m_lab_kill = new common::QLabelEx(this);
            m_lab_kill->setText(ABCDK_GETTEXT("停止:"));

            m_edit_kill = new common::QLineEditEx(this);
            m_edit_kill->setPlaceholderText(ABCDK_GETTEXT("在这里输入应用程序停止(执行)文件, 或双击打开选择对话框."));
            m_edit_kill->setText(m_info->m_kill.c_str());
   
            layout->addWidget(m_lab_kill,3,0);
            layout->addWidget(m_edit_kill,3,1);

            m_lab_rwd = new common::QLabelEx(this);
            m_lab_rwd->setText(ABCDK_GETTEXT("RWD:"));

            m_edit_rwd = new common::QLineEditEx(this);
            m_edit_rwd->setPlaceholderText(ABCDK_GETTEXT("在这里输入应用程序运行RWD, 默认使用当前环境的RWD."));
            m_edit_rwd->setText(m_info->m_rwd.c_str());

            layout->addWidget(m_lab_rwd,4,0);
            layout->addWidget(m_edit_rwd,4,1);

            m_lab_cwd = new common::QLabelEx(this);
            m_lab_cwd->setText(ABCDK_GETTEXT("CWD:"));

            m_edit_cwd = new common::QLineEditEx(this);
            m_edit_cwd->setPlaceholderText(ABCDK_GETTEXT("在这里输入应用程序运行CWD, 默认使用当前环境的CWD."));
            m_edit_cwd->setText(m_info->m_cwd.c_str());

            layout->addWidget(m_lab_cwd,5,0);
            layout->addWidget(m_edit_cwd,5,1);


            m_lab_uid = new common::QLabelEx(this);
            m_lab_uid->setText(ABCDK_GETTEXT("UID:"));

            m_edit_uid = new common::QLineEditEx(this);
            m_edit_uid->setPlaceholderText(common::UtilEx::string_format(ABCDK_GETTEXT("在这里输入应用程序运行UID, 默认使用当前登录的UID(%d)."),getuid()).c_str());
            m_edit_uid->setText(m_info->m_uid.c_str());

            layout->addWidget(m_lab_uid,6,0);
            layout->addWidget(m_edit_uid,6,1);

            m_lab_gid = new common::QLabelEx(this);
            m_lab_gid->setText(ABCDK_GETTEXT("GID:"));

            m_edit_gid = new common::QLineEditEx(this);
            m_edit_gid->setPlaceholderText(common::UtilEx::string_format(ABCDK_GETTEXT("在这里输入应用程序运行GID, 默认使用当前登录的GID(%d)."),getgid()).c_str());
            m_edit_gid->setText(m_info->m_gid.c_str());

            layout->addWidget(m_lab_gid,7,0);
            layout->addWidget(m_edit_gid,7,1);

            m_lab_env = new common::QLabelEx(this);
            m_lab_env->setText(ABCDK_GETTEXT("ENV:"));

            m_edit_env = new common::QPlainTextEditEx(this);
            m_edit_env->setPlaceholderText(ABCDK_GETTEXT("在这里输入应用程序运行ENVs, 默认继承当前环境的ENVs."));
            m_edit_env->setToolTip(ABCDK_GETTEXT("每行一组KEY=VALUE, 输入多组时需要换行."));
            m_edit_env->appendPlainText(m_info->m_env.c_str());

            layout->addWidget(m_lab_env,8,0);
            layout->addWidget(m_edit_env,8,1);

            m_lab_null = new common::QLabelEx(this);
            m_lab_null->setText("");

            m_btn_cancel = new common::QPushButtonEx(this);
            m_btn_cancel->setText(ABCDK_GETTEXT("(&C)取消"));

            m_btn_save = new common::QPushButtonEx(this);
            m_btn_save->setText(ABCDK_GETTEXT("(&S)确定"));

            QHBoxLayout *layout2 = new QHBoxLayout(NULL);
            layout2->setContentsMargins(0,0,0,0);
            layout2->setSpacing(8);

            layout2->addStretch(98);
            layout2->addWidget(m_btn_save,1);
            layout2->addWidget(m_btn_cancel,1);

            layout->addWidget(m_lab_null,9,0);
            layout->addLayout(layout2,9,1);

            connect(m_edit_logo, &common::QLineEditEx::doubleClicked, this, &task_config::onOpenIcon);
            connect(m_edit_exec, &common::QLineEditEx::doubleClicked, this, &task_config::onOpenExec);
            connect(m_edit_kill, &common::QLineEditEx::doubleClicked, this, &task_config::onOpenKill);

            connect(m_btn_cancel,&QPushButton::clicked,this,&task_config::onCancle);
            connect(m_btn_save,&QPushButton::clicked,this,&task_config::onSave);
            
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
