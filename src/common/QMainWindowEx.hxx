/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QMAINWINDOWEX_HXX
#define ABCDK_COMMON_QMAINWINDOWEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT5


namespace abcdk
{
    namespace common
    {
        class QMainWindowEx : public QMainWindow
        {
            Q_OBJECT
        private:
            Qt::Key m_fullscreen_key;

        public:
            QMainWindowEx(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : QMainWindow(parent, flags)
            {
                m_fullscreen_key = Qt::Key::Key_unknown;
            }

            virtual ~QMainWindowEx()
            {
            }

        public:
            void setFullScreenKey(Qt::Key key)
            {
                m_fullscreen_key = key;
            }

        public Q_SLOTS:
            void toggleFullScreen()
            {
                if (isFullScreen())
                    showNormal();
                else
                    showFullScreen();
            }

        Q_SIGNALS:
            void fullScreenStateChanged(bool fullscreen);

        protected:
            virtual void keyPressEvent(QKeyEvent *event)
            {
                if (m_fullscreen_key != Qt::Key_unknown && event->key() == m_fullscreen_key)
                {
                    toggleFullScreen();
                    event->accept(); // 接受事件, 阻止上传.
                }
                else if (event->key() == Qt::Key_Escape && isFullScreen())
                {
                    showNormal();
                    event->accept(); // 接受事件, 阻止上传.
                }
                else
                {
                    QMainWindow::keyPressEvent(event);
                }
            }

            virtual void changeEvent(QEvent *event)
            {
                if (event->type() == QEvent::WindowStateChange)
                {
                    // 仅在最小化时隐藏提示.
                    if (isMinimized())
                        QToolTip::hideText();

                    emit fullScreenStateChanged(isFullScreen());
                }

                QMainWindow::changeEvent(event);
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_COMMON_QMAINWINDOWEX_HXX
