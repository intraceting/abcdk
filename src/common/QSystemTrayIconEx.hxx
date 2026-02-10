/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QSYSTEMTRAYICONEX_HXX
#define ABCDK_COMMON_QSYSTEMTRAYICONEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace common
    {
        class QSystemTrayIconEx : public QSystemTrayIcon
        {
            Q_OBJECT
        private:
        public:
            QSystemTrayIconEx(QObject *parent = nullptr)
                : QSystemTrayIcon(parent)
            {
            }

            QSystemTrayIconEx(const QIcon &icon, QObject *parent = nullptr)
                : QSystemTrayIcon(icon, parent)
            {
            }

            virtual ~QSystemTrayIconEx()
            {
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_COMMON_QSYSTEMTRAYICONEX_HXX
