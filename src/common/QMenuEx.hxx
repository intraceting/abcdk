/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QMENUEX_HXX
#define ABCDK_COMMON_QMENUEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace common
    {
        class QMenuEx : public QMenu
        {
            Q_OBJECT
        private:
        public:
            QMenuEx(QWidget *parent = nullptr)
                : QMenu(parent)
            {
            }

            QMenuEx(const QString &title, QWidget *parent = nullptr)
                : QMenu(title, parent)
            {
            }

            virtual ~QMenuEx()
            {
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_COMMON_QMENUEX_HXX
