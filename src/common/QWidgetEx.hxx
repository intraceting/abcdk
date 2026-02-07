/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_COMMON_QWIDGETEX_HXX
#define ABCDK_COMMON_QWIDGETEX_HXX

#include "Qt.hxx"


#ifdef HAVE_QT


namespace abcdk
{
    namespace common
    {
        class QWidgetEx : public QWidget
        {
            Q_OBJECT
        private:
            bool m_qss_enable;
            bool m_exec_enable;
            QRect m_rect_default;
            QTimer *m_refresh_timer;

        public:
            QWidgetEx(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : QWidget(parent, flags)
            {
                m_qss_enable = false;
                m_exec_enable = false;

                m_refresh_timer = new QTimer(this);
                connect(m_refresh_timer, &QTimer::timeout, this, &QWidgetEx::onRefresh);
            }

            virtual ~QWidgetEx()
            {
                m_refresh_timer->deleteLater();
            }

        public:
            void scaleGeometry(double x_factor, double y_factor)
            {
                if (!m_rect_default.isValid())
                    m_rect_default = geometry();

                setGeometry(m_rect_default.x() * x_factor, m_rect_default.y() * y_factor,
                            m_rect_default.width() * x_factor, m_rect_default.height() * y_factor);
            }

            void enableQSS(bool enable)
            {
                m_qss_enable = enable;
                update();
            }

            void updateProperty(const char *name, const char *value)
            {
                setProperty(name, value);
                style()->unpolish(this);
                style()->polish(this);
                update();
            }

            void startRefresh(int msec)
            {
                if (m_refresh_timer->isActive())
                    m_refresh_timer->stop();

                m_refresh_timer->start(msec);
            }

            void stopRefresh()
            {
                if (m_refresh_timer->isActive())
                    m_refresh_timer->stop();
            }

            int exec(Qt::WindowModality windowModality = Qt::ApplicationModal)
            {
                QEventLoop loop;
                connect(this, &QWidget::destroyed, &loop, &QEventLoop::quit);

                setWindowModality(windowModality);
                show();
                setFocusPolicy(Qt::StrongFocus);
                setFocus();

                m_exec_enable = true;
                int chk = loop.exec(); // 阻塞直到被关闭.
                m_exec_enable = false;

                return chk;
            }

        Q_SIGNALS:
            void clicked();
            void doubleClicked();

        protected:
            virtual void paintEvent(QPaintEvent *event)
            {
                // 先画父窗体.
                QWidget::paintEvent(event);

                if (!m_qss_enable)
                    return;

                QPainter p(this);

                QStyleOption opt;
                opt.init(this);

                // 应用QSS样式.
                style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);

                p.end();
            }

            virtual void onRefresh()
            {
            }

            virtual void mousePressEvent(QMouseEvent *event)
            {
                if (event->button() == Qt::LeftButton)
                {
                    emit clicked();
                }

                QWidget::mousePressEvent(event);
            }

            virtual void mouseDoubleClickEvent(QMouseEvent *event)
            {
                if (event->button() == Qt::LeftButton)
                {
                    emit doubleClicked();
                }

                QWidget::mouseDoubleClickEvent(event);
            }

            virtual void keyPressEvent(QKeyEvent *event)
            {
                if (event->key() == Qt::Key_Escape && m_exec_enable)
                {
                    close();
                    return;
                }

                QWidget::keyPressEvent(event);
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_COMMON_QWIDGETEX_HXX
