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
                std::shared_ptr<task_info> new_info;
                createTab(new_info);
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

        void main_tabview::createTab(std::shared_ptr<task_info> &info)
        {
            if (!info.get())
            {
                info = task_info::newTask(abcdk_time_realtime(6));
                metadata::get()->m_tasks[info->getUUID()] = info; // save to list;
            }

            task_view *new_page = new task_view(info, this);

            info->m_index = addTab(new_page, info->getAppIcon(), info->getAppName()); // add to tabview.

            connect(new_page, &task_view::updateState, this, &main_tabview::updateState);
        }

        void main_tabview::deleteTab(int index)
        {
            task_view *old_page = (task_view *)widget(index);
            if (!old_page)
                return;

            if (old_page->getInfo()->alive())
            {
                QMessageBox::information(this, ABCDK_GETTEXT("提示"), ABCDK_GETTEXT("应用程序正在运行, 不允许删除."));
                return;
            }

            removeTab(index);
            metadata::get()->m_tasks.erase(old_page->getInfo()->getUUID()); // remove from list;
            old_page->deleteLater();
        }

        void main_tabview::detachTab(int index)
        {
            task_view *old_page = (task_view *)widget(index);
            if (!old_page)
                return;

            removeTab(index);
            old_page->getInfo()->m_index = -1;

            task_window *new_win = new task_window(old_page, NULL, window()->windowFlags());
            new_win->resize(800, 500);
            new_win->show();

            connect(new_win, &task_window::detachView, this, &main_tabview::retrieveView);
            connect(old_page, &task_view::updateState, new_win, &task_window::updateState);
        }

        void main_tabview::retrieveView(task_view *view)
        {
            view->getInfo()->m_index = addTab(view, view->getInfo()->getAppIcon(), view->getInfo()->getAppName());
        }

        void main_tabview::updateState(std::shared_ptr<task_info> &info)
        {
            if (info->m_index < 0)
                return;

            setTabText(info->m_index, info->getAppName());
            setTabIcon(info->m_index, info->getAppIcon());
        }

        void main_tabview::deInit()
        {
        }

        void main_tabview::Init()
        {
            metadata::get()->loadTasks();

            if (metadata::get()->m_tasks.size() <= 0)
            {
                std::shared_ptr<task_info> new_info;
                createTab(new_info);
            }
            else
            {
                std::vector<std::shared_ptr<task_info>> tab_pages(metadata::get()->m_tasks.size());
                for(auto &one:metadata::get()->m_tasks)
                {
                    if(one.second->m_index >= 0)
                    {
                        tab_pages[one.second->m_index] = one.second;
                    }
                    else 
                    {

                    }
                }

            }

            connect(this, &main_tabview::clickedRight, this, &main_tabview::showRightClickMenu);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT5
