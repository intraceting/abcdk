/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_TASK_CONFIG_HXX
#define ABCDK_LAUNCHER_TASK_CONFIG_HXX

#include "abcdk.h"
#include "../common/QWidgetEx.hxx"
#include "metadata.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        class task_config : public common::QWidgetEx
        {
            Q_OBJECT
        private:
        public:
            task_config(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : common::QWidgetEx(parent, flags)
            {
                Init();
            }

            virtual ~task_config()
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

#endif // ABCDK_LAUNCHER_TASK_CONFIG_HXX
