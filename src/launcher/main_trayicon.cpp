/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "main_trayicon.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        void main_trayicon::deInit()
        {

        }

        void main_trayicon::Init()
        {
            setIcon(common::QUtilEx::getIcon(":/images/logo-v1.png"));
            setToolTip(ABCDK_GETTEXT("应用程序启动器"));

            common::QMenuEx *menu = new common::QMenuEx;

            QAction *actionShow = new QAction(ABCDK_GETTEXT("主窗体"));
            QAction *actionAbout = new QAction(ABCDK_GETTEXT("关于"));
            QAction *actionQuit = new QAction(ABCDK_GETTEXT("退出"));

            menu->addAction(actionShow);
            menu->addSeparator();
            menu->addAction(actionAbout);
            menu->addAction(actionQuit);

            setContextMenu(menu);

            connect(actionShow, &QAction::triggered, this, &main_trayicon::onShow);
            connect(actionAbout, &QAction::triggered, this, &main_trayicon::onAbout);
            connect(actionQuit, &QAction::triggered, this, &main_trayicon::onQuit);
        }
    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5