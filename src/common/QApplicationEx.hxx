/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_COMMON_QAPPLICATIONEX_HXX
#define ABCDK_COMMON_QAPPLICATIONEX_HXX

#include "Qt.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace common
    {
        class QApplicationEx : public QApplication
        {
            Q_OBJECT
        private:
        public:
            QApplicationEx(int &argc, char *argv[])
                : QApplication(argc, argv)
            {
            }

            virtual ~QApplicationEx()
            {
            }
        };

    } // namespace common
} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_COMMON_QAPPLICATIONEX_HXX
