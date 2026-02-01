/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_TASK_VIEW_PART3_HXX
#define ABCDK_LAUNCHER_TASK_VIEW_PART3_HXX

#include "abcdk.h"
#include "../common/QPlainTextEditEx.hxx"
#include "../common/QTabWidgetEx.hxx"
#include "metadata.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        class task_view_part3 : public common::QTabWidgetEx
        {
            Q_OBJECT
        private:
            common::QPlainTextEditEx *m_edit_stdout;
            common::QPlainTextEditEx *m_edit_stderr;
        public:
            task_view_part3(QWidget *parent = nullptr)
                : common::QTabWidgetEx(parent)
            {
                Init();
            }

            virtual ~task_view_part3()
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

#endif // ABCDK_LAUNCHER_TASK_VIEW_PART3_HXX
