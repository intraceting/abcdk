/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "main_window.hxx"
#include "application.hxx"

int main(int argc, char *argv[])
{
    int chk = 0;

#ifdef HAVE_QT5

    abcdk::launcher::metadata::get()->parseCmdLine(argc,argv);

    if(abcdk::launcher::metadata::get()->isPrintUsage())
    {
        abcdk::launcher::metadata::get()->printUsage();
        return 0;
    }

    abcdk_locale_setup(NULL, NULL, NULL);

    abcdk::launcher::application a(argc, argv);

    QApplication::setQuitOnLastWindowClosed(false);
    
    abcdk::launcher::main_window main_win;
    main_win.resize(800, 600);
    main_win.show();

    chk = a.exec();

#else // #ifdef HAVE_QT5
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含Qt工具."));
#endif // #ifdef HAVE_QT5

    return abs(chk);
}
