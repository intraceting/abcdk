/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QSCROLLAREAEX_HXX
#define ABCDK_COMMON_QSCROLLAREAEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT


namespace abcdk
{
    namespace common
    {
        class QScrollAreaEx : public QScrollArea
        {
            Q_OBJECT
        private:
            QRect m_rect_default;
            bool m_mouse_is_inside;

        public:
            QScrollAreaEx(QWidget *parent = nullptr)
                : QScrollArea(parent)
            {
                //
                setMouseTracking(true);
                // 可视区.
                viewport()->installEventFilter(this);
                // 滚动条.
                verticalScrollBar()->installEventFilter(this);
                horizontalScrollBar()->installEventFilter(this);

                m_mouse_is_inside = false;
            }

            virtual ~QScrollAreaEx()
            {
            }

        public:
            void scaleGeometry(double x_factor, double y_factor)
            {
                if (!m_rect_default.isValid())
                    m_rect_default = geometry();

                setGeometry(m_rect_default.x() * x_factor, m_rect_default.y() * y_factor,
                            m_rect_default.width() * x_factor, m_rect_default.height() * y_factor);
            }

            bool mouseIsInside()
            {
                return m_mouse_is_inside;
            }

        protected:
            virtual bool eventFilter(QObject *obj, QEvent *event)
            {
                Q_UNUSED(obj);

                if (event->type() == QEvent::Enter)
                {
                    m_mouse_is_inside = true;
                }
                else if (event->type() == QEvent::Leave)
                {
                    m_mouse_is_inside = false;
                }

                return QScrollArea::eventFilter(obj, event);
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_COMMON_QSCROLLAREAEX_HXX