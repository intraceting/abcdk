/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QTABWIDGETEX_HXX
#define ABCDK_COMMON_QTABWIDGETEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace common
    {
        class QTabWidgetEx : public QTabWidget
        {
            Q_OBJECT
        private:
            QRect m_rect_default;
            QTimer *m_refresh_timer;

        public:
            QTabWidgetEx(QWidget *parent = nullptr)
                : QTabWidget(parent)
            {
                m_refresh_timer = new QTimer(this);
                connect(m_refresh_timer, &QTimer::timeout, this, &QTabWidgetEx::onRefresh);
            }

            virtual ~QTabWidgetEx()
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

        Q_SIGNALS:
            void clickedRight(int index, const QPoint &globalPos);

        protected:
            virtual void onRefresh()
            {
            }

            virtual void mousePressEvent(QMouseEvent *event)
            {
                if (event->button() == Qt::RightButton)
                {
                    int index = tabBar()->tabAt(event->pos());
                    emit clickedRight(index, event->globalPos());
                }

                QTabWidget::mousePressEvent(event);
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_COMMON_QTABWIDGETEX_HXX
