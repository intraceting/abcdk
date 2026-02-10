/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "main_window.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        void main_window::setStyleSheet(size_t idx)
        {
            static std::vector<std::string> qss_list = {":/qss/null.qss", ":/qss/border.qss"};

            m_app->setStyleSheet(common::QUtilEx::loadStyleSheet(qss_list[idx % qss_list.size()]));
        }

        void main_window::onShow()
        {
            show();
        }

        void main_window::onAbout()
        {
            QMessageBox::about(this, ABCDK_GETTEXT("关于"), ABCDK_GETTEXT("应用程序启动器"));
        }

        void main_window::onQuit()
        {
            close();
        }

        void main_window::deInit()
        {
        }

        void main_window::Init()
        {
            m_app = qobject_cast<QApplication *>(QApplication::instance());

            abcdk_trace_printf(LOG_INFO, "Qt GUI Platform: %s", QGuiApplication::platformName().toStdString().c_str());

            setObjectName("main_window");
            setWindowTitle(ABCDK_GETTEXT("应用程序启动器"));
            setFullScreenKey(Qt::Key_F11);

            m_tabview = new main_tabview(this);
            setCentralWidget(m_tabview);

            setStyleSheet(1);

            if (!QSystemTrayIcon::isSystemTrayAvailable())
            {
                abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("不支持注册托盘图标."));
                return;
            }

            main_trayicon *tray = new main_trayicon(this);
            tray->show();

            // 关联托盘事件.
            QObject::connect(tray, &main_trayicon::onShow, this, &main_window::onShow);
            QObject::connect(tray, &main_trayicon::onAbout, this, &main_window::onAbout);
            QObject::connect(tray, &main_trayicon::onQuit, this, &main_window::onQuit);
        }

        void main_window::closeEvent(QCloseEvent *event)
        {
            if (abcdk_atomic_load(&metadata::get()->m_alive_tasks_count) > 0)
            {
                QMessageBox::information(this, ABCDK_GETTEXT("提示"), ABCDK_GETTEXT("还有应用程序正在运行, 主窗体将最小化到托盘."));

                hide();          // 隐藏窗体.
                event->ignore(); // 阻止默认关闭.
            }
            else
            {
                event->accept();
                m_app->exit();
            }
        }
    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
