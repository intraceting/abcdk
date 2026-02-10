/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_TASK_CONFIG_HXX
#define ABCDK_LAUNCHER_TASK_CONFIG_HXX

#include "abcdk.h"
#include "../common/QDialogEx.hxx"
#include "../common/QWidgetEx.hxx"
#include "../common/QPushButtonEx.hxx"
#include "../common/QLineEditEx.hxx"
#include "../common/QPlainTextEditEx.hxx"
#include "../common/QLabelEx.hxx"
#include "../common/UtilEx.hxx"
#include "metadata.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        class task_config : public common::QDialogEx
        {
            Q_OBJECT
        private:
            std::shared_ptr<task_info> m_info;

            common::QLabelEx *m_lab_name;
            common::QLineEditEx *m_edit_name;
            common::QLabelEx *m_lab_logo;
            common::QLineEditEx *m_edit_logo;
            common::QLabelEx *m_lab_exec;
            common::QLineEditEx *m_edit_exec;
            common::QLabelEx *m_lab_kill;
            common::QLineEditEx *m_edit_kill;
            common::QLabelEx *m_lab_rwd;
            common::QLineEditEx *m_edit_rwd;
            common::QLabelEx *m_lab_cwd;
            common::QLineEditEx *m_edit_cwd;
            common::QLabelEx *m_lab_uid;
            common::QLineEditEx *m_edit_uid;
            common::QLabelEx *m_lab_gid;
            common::QLineEditEx *m_edit_gid;
            common::QLabelEx *m_lab_env;
            common::QPlainTextEditEx *m_edit_env;
            common::QLabelEx *m_lab_null;
            common::QPushButtonEx *m_btn_cancel;
            common::QPushButtonEx *m_btn_save;
        public:
            task_config(std::shared_ptr<task_info> &info, QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : common::QDialogEx(parent, flags)
            {
                Init(info);
            }

            virtual ~task_config()
            {
                deInit();
            }
        private Q_SLOTS:
            void onOpenIcon();
            void onOpenExec();
            void onOpenKill();
            void onCancle();
            void onSave();
        protected:
            void deInit();
            void Init(std::shared_ptr<task_info> &info);
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_LAUNCHER_TASK_CONFIG_HXX
