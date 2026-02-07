/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_MAIN_WINDOW_HXX
#define ABCDK_LAUNCHER_MAIN_WINDOW_HXX

#include "abcdk.h"
#include "../common/QUtilEx.hxx"
#include "../common/QMainWindowEx.hxx"
#include "main_tabview.hxx"
#include "main_trayicon.hxx"


#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        class main_window : public common::QMainWindowEx
        {
            Q_OBJECT
        private:
            QApplication *m_app;
            main_tabview *m_tabview;
        public:
            main_window(QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags())
                : common::QMainWindowEx(parent, f)
            {
                Init();
            }

            virtual ~main_window()
            {
                deInit();
            }
        public:
            void setStyleSheet(size_t idx);
        private slots:
            void onShow();
            void onAbout();
            void onQuit();
        protected:
            void deInit();
            void Init();
            virtual void closeEvent(QCloseEvent *event);
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_LAUNCHER_MAIN_WINDOW_HXX
