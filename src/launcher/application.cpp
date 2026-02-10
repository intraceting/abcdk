/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "application.hxx"


#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        void application::deInit()
        {

        }

        void application::Init(int &argc, char *argv[])
        {
            ABCDK_UNUSED(argc);
            ABCDK_UNUSED(argv);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
