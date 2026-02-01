/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QPUSHBUTTONEX_HXX
#define ABCDK_COMMON_QPUSHBUTTONEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT


namespace abcdk
{
    namespace common
    {
        class QPushButtonEx : public QPushButton
        {
            Q_OBJECT
        private:
            QRect m_rect_default;

        public:
            QPushButtonEx(QWidget *parent = nullptr)
                : QPushButton(parent)
            {
            }

            virtual ~QPushButtonEx()
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
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_COMMON_QPUSHBUTTONEX_HXX
