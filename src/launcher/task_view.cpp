/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "task_view.hxx"

#ifdef HAVE_QT

namespace abcdk
{
    namespace launcher
    {
        void task_view::deInit()
        {
        }

        void task_view::Init()
        {
            QVBoxLayout *layout = new QVBoxLayout(this);
            layout->setContentsMargins(10,10,10,10);
            layout->setSpacing(8);

            task_view_part1 *part1 = new task_view_part1(this);

            task_view_part2 *part2 = new task_view_part2(this);

            task_view_part3 *part3 = new task_view_part3(this);

            layout->addWidget(part1);
            layout->addWidget(part3);
            layout->addWidget(part2);
        }

        void task_view::mousePressEvent(QMouseEvent *event)
        {
            if (event->button() == Qt::RightButton)
            {
                event->accept();//拦截鼠标右键单击事件,以防止传递给父窗体.
                return;
            }

            common::QWidgetEx::mousePressEvent(event);
        }

    } // namespace launcher

} // namespace abcdk

#endif // #ifdef HAVE_QT
