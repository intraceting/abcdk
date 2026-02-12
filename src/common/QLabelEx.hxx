/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QLABELEX_HXX
#define ABCDK_COMMON_QLABELEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace common
    {
        class QLabelEx : public QLabel
        {
            Q_OBJECT
        private:
            QRect m_rect_default;
            QTimer *m_refresh_timer;

        public:
            QLabelEx(QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags())
                : QLabel(parent, flags)
            {
                m_refresh_timer = new QTimer(this);
                connect(m_refresh_timer, &QTimer::timeout, this, &QLabelEx::onRefresh);
            }

            virtual ~QLabelEx()
            {
                m_refresh_timer->deleteLater();
            }

        public:
            void updateProperty(const char *name, const char *value)
            {
                setProperty(name, value);
                style()->unpolish(this);
                style()->polish(this);
                update();
            }

            void scaleGeometry(double x_factor, double y_factor)
            {
                if (!m_rect_default.isValid())
                    m_rect_default = geometry();

                setGeometry(m_rect_default.x() * x_factor, m_rect_default.y() * y_factor,
                            m_rect_default.width() * x_factor, m_rect_default.height() * y_factor);
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

        Q_SIGNALS:
            void clicked();
            void doubleClicked();

        protected:
            virtual void onRefresh()
            {
            }

            virtual void mousePressEvent(QMouseEvent *event)
            {
                if (event->button() == Qt::LeftButton)
                {
                    emit clicked();
                }

                QLabel::mousePressEvent(event);
            }

            virtual void mouseDoubleClickEvent(QMouseEvent *event)
            {
                if (event->button() == Qt::LeftButton)
                {
                    emit doubleClicked();
                }

                QLabel::mouseDoubleClickEvent(event);
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_COMMON_QLABELEX_HXX