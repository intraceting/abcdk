/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_MAIN_TABVIEW_HXX
#define ABCDK_LAUNCHER_MAIN_TABVIEW_HXX

#include "abcdk.h"
#include "../common/QMenuEx.hxx"
#include "../common/QMainWindowEx.hxx"
#include "../common/QTabWidgetEx.hxx"
#include "task_view.hxx"
#include "task_window.hxx"


#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        class main_tabview : public common::QTabWidgetEx
        {
            Q_OBJECT
        private:
        public:
            main_tabview(QWidget *parent = nullptr)
                : common::QTabWidgetEx(parent)
            {
                Init();
            }

            virtual ~main_tabview()
            {
                deInit();
            }

        private Q_SLOTS:
            void showRightClickMenu(int index, const QPoint &globalPos);
            void createTab();
            void deleteTab(int index);
            void detachTab(int index);
            void retrieveView(task_view *view);
            void updateState(std::shared_ptr<task_info> &info);
        protected:
            void deInit();
            void Init();
            void reLoad();
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_LAUNCHER_MAIN_TABVIEW_HXX
