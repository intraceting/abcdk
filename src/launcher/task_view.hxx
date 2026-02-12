/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_TASK_VIEW_HXX
#define ABCDK_LAUNCHER_TASK_VIEW_HXX

#include "abcdk.h"
#include "../common/QWidgetEx.hxx"
#include "../common/QTabWidgetEx.hxx"
#include "../common/QCheckBoxEx.hxx"
#include "../common/QPlainTextEditEx.hxx"
#include "../common/QPushButtonEx.hxx"
#include "../common/QUtilEx.hxx"
#include "metadata.hxx"
#include "task_config.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        class task_view : public common::QWidgetEx
        {
            Q_OBJECT
        private:
            std::shared_ptr<task_info> m_info;

            common::QLineEditEx *m_edit_exec;
            common::QPushButtonEx *m_btn_conf;

            common::QPushButtonEx *m_btn_clear;
            common::QCheckBoxEx *m_chk_autoroll;
            common::QPushButtonEx *m_btn_start;
            common::QPushButtonEx *m_btn_stop;
            
            common::QPlainTextEditEx *m_edit_stdout;
            common::QPlainTextEditEx *m_edit_stderr;

            int m_stop_tip;
            
        public:
            task_view(std::shared_ptr<task_info> &info, QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : common::QWidgetEx(parent, flags)
            {
                Init(info);
            }

            virtual ~task_view()
            {
                deInit();
            }
        public:
            std::shared_ptr<task_info> getInfo();
        private Q_SLOTS:
            void onEditExecChanged();
            void onPopConfig();
            void onClear();
            void onStart();
            void onStop();
        Q_SIGNALS:
            void updateState(std::shared_ptr<task_info> &info);
        protected:
            void deInit();
            void Init(std::shared_ptr<task_info> &info);
            virtual void mousePressEvent(QMouseEvent *event);
            virtual void onRefresh();
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_LAUNCHER_TASK_VIEW_HXX
