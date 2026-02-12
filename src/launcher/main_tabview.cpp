/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "main_tabview.hxx"

#ifdef HAVE_QT5

namespace abcdk
{
    namespace launcher
    {
        void main_tabview::showRightClickMenu(int index, const QPoint &globalPos)
        {
            common::QMenuEx menu;

            QAction *createAction = menu.addAction(ABCDK_GETTEXT("(&N)新建"));
            QAction *separatorAction1 = menu.addSeparator();
            QAction *deleteAction = menu.addAction(ABCDK_GETTEXT("(&D)删除"));
            QAction *detachAction = menu.addAction(ABCDK_GETTEXT("(&M)移动到独立窗体"));

            if (index < 0)
            {
                deleteAction->setEnabled(false);
                detachAction->setEnabled(false);
            }

            QAction *action = menu.exec(globalPos);
            if (!action)
                return;

            if (action == createAction)
            {
                createTab();
            }
            else if (action == deleteAction)
            {
                deleteTab(index);
            }
            else if (action == detachAction)
            {
                detachTab(index);
            }
        }

        void main_tabview::createTab()
        {
            std::shared_ptr<task_info> new_task = task_info::newTask(time(NULL));
            metadata::get()->m_tasks[new_task->uuid()] = new_task; // save to list;

            task_view *new_page = new task_view(new_task, this);

            new_task->m_tab_index = addTab(new_page, new_task->getAppIcon(), new_task->getAppName()); // add to tabview.

            connect(new_page, &task_view::updateState, this, &main_tabview::updateState);
        }

        void main_tabview::deleteTab(int index)
        {
            task_view *old_page = (task_view *)widget(index);
            if (!old_page)
                return;

            if (old_page->getInfo()->alive())
            {
                QMessageBox::information(this, ABCDK_GETTEXT("提示"), ABCDK_GETTEXT("不允许删除, 应用程序正在运行."));
                return;
            }

            removeTab(index);
            metadata::get()->m_tasks.erase(old_page->getInfo()->uuid()); // remove from list;
            old_page->deleteLater();
        }

        void main_tabview::detachTab(int index)
        {
            task_view *old_page = (task_view *)widget(index);
            if (!old_page)
                return;

            removeTab(index);
            old_page->getInfo()->m_tab_index = -1;

            task_window *new_win = new task_window(old_page, NULL, window()->windowFlags());
            new_win->resize(600, 400);
            new_win->show();

            connect(new_win, &task_window::detachView, this, &main_tabview::retrieveView);
            connect(old_page, &task_view::updateState, new_win, &task_window::updateState);
        }

        void main_tabview::retrieveView(task_view *view)
        {
            view->getInfo()->m_tab_index = addTab(view, view->getInfo()->getAppIcon(), view->getInfo()->getAppName());
        }

        void main_tabview::updateState(std::shared_ptr<task_info> &info)
        {
            if (info->m_tab_index < 0)
                return;

            setTabText(info->m_tab_index, info->getAppName());
            setTabIcon(info->m_tab_index, info->getAppIcon());
        }

        void main_tabview::deInit()
        {
        }

        void main_tabview::Init()
        {
            createTab();

            connect(this, &main_tabview::clickedRight, this, &main_tabview::showRightClickMenu);
        }


    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
