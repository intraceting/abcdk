/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "metadata.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        std::shared_ptr<metadata> metadata::get()
        {
            static std::shared_ptr<metadata> only_one = std::shared_ptr<metadata>(new metadata, [](void *p)
                                                                                  {if(p){delete (metadata*)p;} });

            return only_one;
        }

        void metadata::Init(int &argc, char *argv[])
        {
            m_args = abcdk_option_alloc("--");
            if (!m_args)
                return;

            abcdk_getargs(m_args, argc, argv);
            argc = 1; // 只保留第一个参数.

            abcdk_locale_setup(NULL, NULL, NULL);
        }

        void metadata::PrintUsage(FILE *out /*= stderr*/)
        {
            fprintf(out, "\n描述:\n");

            fprintf(out, "\n\t简单应用程序启动器.\n");
        }

        bool metadata::IsPrintUsage()
        {
            return (abcdk_option_exist(m_args, "--help"));
        }

        void metadata::deInit()
        {
            abcdk_option_free(&m_args);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
