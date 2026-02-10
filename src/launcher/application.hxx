/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_APPLICATION_HXX
#define ABCDK_LAUNCHER_APPLICATION_HXX

#include "abcdk.h"
#include "../common/QApplicationEx.hxx"
#include "metadata.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        class application : public common::QApplicationEx
        {
            Q_OBJECT
        private:
        public:
            application(int &argc, char *argv[])
                : common::QApplicationEx(argc, argv)
            {
                Init(argc,argv);
            }

            virtual ~application()
            {

            }

        protected:
            void deInit();
            void Init(int &argc, char *argv[]);
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5

#endif // ABCDK_LAUNCHER_APPLICATION_HXX
