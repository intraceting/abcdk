/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_TASK_VIEW_PART1_HXX
#define ABCDK_LAUNCHER_TASK_VIEW_PART1_HXX

#include "abcdk.h"
#include "../common/QLineEditEx.hxx"
#include "../common/QPushButtonEx.hxx"
#include "../common/QWidgetEx.hxx"
#include "metadata.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        class task_view_part1 : public common::QWidgetEx
        {
            Q_OBJECT
        private:
            common::QLineEditEx *m_edit_cmd;
            common::QPushButtonEx *m_btn_conf;
        public:
            task_view_part1(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : common::QWidgetEx(parent, flags)
            {
                Init();
            }

            virtual ~task_view_part1()
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

#endif // ABCDK_LAUNCHER_TASK_VIEW_PART1_HXX
