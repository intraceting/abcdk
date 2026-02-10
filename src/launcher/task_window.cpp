/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_window.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        void task_window::updateState(std::shared_ptr<task_info> &info)
        {
            task_view *view = (task_view *)centralWidget();

            if (abcdk_strcmp(view->getInfo()->uuid(), info->uuid(), 0) != 0)
                return;

            setWindowTitle(info->getAppName());
            setWindowIcon(info->getAppIcon());
        }

        void task_window::deInit()
        {
            
        }

        void task_window::Init(task_view *view)
        {
            setObjectName("task_window");
            setAttribute(Qt::WA_DeleteOnClose);

            view->setParent(this);
            view->show();

            setCentralWidget(view);

            setWindowTitle(view->getInfo()->getAppName());
            setWindowIcon(view->getInfo()->getAppIcon());
        }

        void task_window::closeEvent(QCloseEvent *event)
        {
            task_view *view = (task_view*)takeCentralWidget();//解除关系.

            emit detachView(view); // 通知解除关系.
            event->accept(); // 不在传递.
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
