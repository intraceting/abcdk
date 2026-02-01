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
#include "metadata.hxx"
#include "task_view_part1.hxx"
#include "task_view_part2.hxx"
#include "task_view_part3.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        class task_view : public common::QWidgetEx
        {
            Q_OBJECT
        private:
        public:
            task_view(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : common::QWidgetEx(parent, flags)
            {
                Init();
            }

            virtual ~task_view()
            {
                deInit();
            }

        protected:
            void deInit();
            void Init();
            virtual void mousePressEvent(QMouseEvent *event);
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_LAUNCHER_TASK_VIEW_HXX
