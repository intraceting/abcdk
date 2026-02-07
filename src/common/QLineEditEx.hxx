/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QLINEEDITEX_HXX
#define ABCDK_COMMON_QLINEEDITEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT


namespace abcdk
{
    namespace common
    {
        class QLineEditEx : public QLineEdit
        {
            Q_OBJECT
        private:
            QRect m_rect_default;

        public:
            QLineEditEx(QWidget *parent = nullptr)
                : QLineEdit(parent)
            {
            }

            virtual ~QLineEditEx()
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
        Q_SIGNALS:
            void clicked();
            void doubleClicked();

        protected:
            virtual void mousePressEvent(QMouseEvent *event)
            {
                if (event->button() == Qt::LeftButton)
                {
                    emit clicked();
                }

                QLineEdit::mousePressEvent(event);
            }

            virtual void mouseDoubleClickEvent(QMouseEvent *event)
            {
                if (event->button() == Qt::LeftButton)
                {
                    emit doubleClicked();
                }

                QLineEdit::mouseDoubleClickEvent(event);
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_COMMON_QLINEEDITEX_HXX
