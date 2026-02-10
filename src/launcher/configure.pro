TARGET = abcdk-launcher
TEMPLATE = app

DESTDIR = $$BUILD_PATH

DEFINES += HAVE_QT5

INCLUDEPATH += $$PWD/../common/

HEADERS += $$PWD/../common/QApplicationEx.hxx
HEADERS += $$PWD/../common/QCheckBoxEX.hxx
HEADERS += $$PWD/../common/QDialogEx.hxx
HEADERS += $$PWD/../common/QLabelEx.hxx
HEADERS += $$PWD/../common/QLineEditEx.hxx
HEADERS += $$PWD/../common/QMainWindowEx.hxx
HEADERS += $$PWD/../common/QMenuEx.hxx
HEADERS += $$PWD/../common/QObjectEx.hxx
HEADERS += $$PWD/../common/QPlainTextEditEx.hxx
HEADERS += $$PWD/../common/QPushButtonEx.hxx
HEADERS += $$PWD/../common/QScrollAreaEx.hxx
HEADERS += $$PWD/../common/QSystemTrayIconEx.hxx
HEADERS += $$PWD/../common/QTabWidgetEx.hxx
HEADERS += $$PWD/../common/QUtilEx.hxx
HEADERS += $$PWD/../common/QWidgetEx.hxx
HEADERS += $$PWD/../common/UtilEx.hxx
RESOURCES += $$PWD/resource/resources.qrc
SOURCES += $$PWD/application.cpp
HEADERS += $$PWD/application.hxx
SOURCES += $$PWD/main_tabview.cpp
HEADERS += $$PWD/main_tabview.hxx
SOURCES += $$PWD/main_trayicon.cpp
HEADERS += $$PWD/main_trayicon.hxx
SOURCES += $$PWD/main_window.cpp
HEADERS += $$PWD/main_window.hxx
SOURCES += $$PWD/main.cpp
SOURCES += $$PWD/metadata.cpp
HEADERS += $$PWD/metadata.hxx
SOURCES += $$PWD/task_config.cpp
HEADERS += $$PWD/task_config.hxx
SOURCES += $$PWD/task_info.cpp
HEADERS += $$PWD/task_info.hxx
SOURCES += $$PWD/task_view.cpp
HEADERS += $$PWD/task_view.hxx
SOURCES += $$PWD/task_window.cpp
HEADERS += $$PWD/task_window.hxx

QT += core gui widgets svg

#try to disable RPATH
CONFIG += no_qt_rpath

#try to disable RPATH
#QMAKE_LFLAGS_RPATH =
#QMAKE_LFLAGS_RPATHLINK =



