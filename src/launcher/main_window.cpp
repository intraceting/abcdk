/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "main_window.hxx"


#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        void main_window::deInit()
        {
        }

        void main_window::Init()
        {
            m_app = qobject_cast<QApplication *>(QApplication::instance());

            abcdk_trace_printf(LOG_INFO, "Qt GUI Platform: %s", QGuiApplication::platformName().toStdString().c_str());

            setWindowTitle(ABCDK_GETTEXT("应用程序启动器"));
            setFullScreenKey(Qt::Key_F11);

            m_tabview = new main_tabview(this);
            setCentralWidget(m_tabview);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
