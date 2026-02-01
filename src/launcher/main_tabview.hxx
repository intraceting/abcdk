/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_MAIN_TABVIEW_HXX
#define ABCDK_LAUNCHER_MAIN_TABVIEW_HXX

#include "abcdk.h"
#include "../common/QTabWidgetEx.hxx"
#include "task_view.hxx"


#ifdef HAVE_QT

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

        public Q_SLOTS:
            void showRightClickMenu(int index, const QPoint &globalPos);

        protected:
            void deInit();
            void Init();
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_LAUNCHER_MAIN_TABVIEW_HXX
