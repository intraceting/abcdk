/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_MAIN_TRAYICON_HXX
#define ABCDK_LAUNCHER_MAIN_TRAYICON_HXX

#include "abcdk.h"
#include "../common/QSystemTrayIconEx.hxx"
#include "../common/QMenuEx.hxx"
#include "../common/QUtilEx.hxx"


#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        class main_trayicon : public common::QSystemTrayIconEx
        {
            Q_OBJECT
        private:
        public:
            main_trayicon(QWidget *parent = nullptr)
                : common::QSystemTrayIconEx(parent)
            {
                Init();
            }

            virtual ~main_trayicon()
            {
                deInit();
            }
        
        signals:
            void onShow();
            void onAbout();
            void onQuit();
        protected:
            void deInit();
            void Init();
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_LAUNCHER_MAIN_TRAYICON_HXX
