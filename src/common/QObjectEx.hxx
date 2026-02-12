/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QOBJECTEX_HXX
#define ABCDK_COMMON_QOBJECTEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace common
    {
        class QObjectEx : public QObject
        {
            Q_OBJECT
        private:
        public:
            QObjectEx(QObject *parent=nullptr)
                : QObject(parent)
            {
            }

            virtual ~QObjectEx()
            {
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_COMMON_QOBJECTEX_HXX
