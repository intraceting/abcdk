/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LAUNCHER_METADATA_HXX
#define ABCDK_LAUNCHER_METADATA_HXX

#include "abcdk.h"
#include "QObjectEx.hxx"
#include <iostream>
#include <memory>

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        class metadata :public common::QObjectEx
        {
            Q_OBJECT
        private:
            abcdk_option_t *m_args;
        protected:
            metadata(QObject *parent=nullptr)
                :common::QObjectEx(parent)
            {
                m_args = NULL;
            }

            virtual ~metadata()
            {
                deInit();
            }

        public:
            static std::shared_ptr<metadata> get();
        public:
            void Init(int &argc, char *argv[]);
            void PrintUsage(FILE *out = stderr);
            bool IsPrintUsage();
        protected:
            void deInit();
            
        };

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT

#endif // ABCDK_LAUNCHER_METADATA_HXX
