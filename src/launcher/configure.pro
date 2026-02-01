TARGET = abcdk-launcher
TEMPLATE = app

DEFINES += HAVE_QT

INCLUDEPATH += $$PWD/../common/

HEADERS += $$PWD/../common/QApplicationEx.hxx
HEADERS += $$PWD/../common/QCheckBoxEX.hxx
HEADERS += $$PWD/../common/QLabelEx.hxx
HEADERS += $$PWD/../common/QLineEditEx.hxx
HEADERS += $$PWD/../common/QMainWindowEx.hxx
HEADERS += $$PWD/../common/QObjectEx.hxx
HEADERS += $$PWD/../common/QPlainTextEditEx.hxx
HEADERS += $$PWD/../common/QPushButtonEx.hxx
HEADERS += $$PWD/../common/QScrollAreaEx.hxx
HEADERS += $$PWD/../common/QTabWidgetEx.hxx
HEADERS += $$PWD/../common/QUtilEx.hxx
HEADERS += $$PWD/../common/QWidgetEx.hxx
RESOURCES += $$PWD/resource/resources.qrc
SOURCES += $$PWD/application.cpp
HEADERS += $$PWD/application.hxx
SOURCES += $$PWD/main_tabview.cpp
HEADERS += $$PWD/main_tabview.hxx
SOURCES += $$PWD/main_window.cpp
HEADERS += $$PWD/main_window.hxx
SOURCES += $$PWD/main.cpp
SOURCES += $$PWD/metadata.cpp
HEADERS += $$PWD/metadata.hxx
SOURCES += $$PWD/task_config.cpp
HEADERS += $$PWD/task_config.hxx
SOURCES += $$PWD/task_view_part1.cpp
HEADERS += $$PWD/task_view_part1.hxx
SOURCES += $$PWD/task_view_part2.cpp
HEADERS += $$PWD/task_view_part2.hxx
SOURCES += $$PWD/task_view_part3.cpp
HEADERS += $$PWD/task_view_part3.hxx
SOURCES += $$PWD/task_view.cpp
HEADERS += $$PWD/task_view.hxx
SOURCES += $$PWD/task_window.cpp
HEADERS += $$PWD/task_window.hxx

QT += core gui widgets svg

DESTDIR = $$BUILD_PATH


