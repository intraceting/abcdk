/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "application.hxx"


#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        void application::deInit()
        {

        }

        void application::Init(int &argc, char *argv[])
        {
            Q_UNUSED(argc);
            Q_UNUSED(argv);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
