/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "main_tabview.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        void main_tabview::showRightClickMenu(int index, const QPoint &globalPos)
        {
            
            task_view *new_page = new task_view(this);

            
            addTab(new_page, "第一页");
        }

        void main_tabview::deInit()
        {
        }

        void main_tabview::Init()
        {
            connect(this, &main_tabview::clickedRight, this, &main_tabview::showRightClickMenu);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
