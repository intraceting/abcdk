/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_TASK_WINDOW_HXX
#define ABCDK_LAUNCHER_TASK_WINDOW_HXX

#include "abcdk.h"
#include "../common/QMainWindowEx.hxx"
#include "metadata.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        class task_window : public common::QMainWindowEx
        {
            Q_OBJECT
        private:
        public:
            task_window(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : common::QMainWindowEx(parent, flags)
            {
                Init();
            }

            virtual ~task_window()
            {
                deInit();
            }

        protected:
            void deInit();
            void Init();
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_LAUNCHER_TASK_WINDOW_HXX
