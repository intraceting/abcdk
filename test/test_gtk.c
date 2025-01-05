/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "entry.h"

#ifdef HAVE_GTK
#include <gtk/gtk.h>

// 按钮点击回调函数
static void on_button_clicked(GtkWidget *widget, gpointer data) {
    g_print("按钮被点击了！\n");
}

int abcdk_test_gtk(abcdk_option_t *args)
{
    // 初始化GTK
    gtk_init(NULL,NULL);

    // 创建主窗口
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "GTK例子");
    gtk_window_set_default_size(GTK_WINDOW(window), 400, 200);
    
    // 设置窗口关闭事件
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    // 创建一个按钮
    GtkWidget *button = gtk_button_new_with_label("点击我");
      gtk_widget_set_margin_start(button, 10);  // 左边距
    gtk_widget_set_margin_end(button, 10);    // 右边距
    gtk_widget_set_margin_top(button, 10);    // 上边距
    gtk_widget_set_margin_bottom(button, 10); // 下边距
    // 连接按钮点击信号
    g_signal_connect(button, "clicked", G_CALLBACK(on_button_clicked), NULL);

    // 将按钮添加到窗口中
    gtk_container_add(GTK_CONTAINER(window), button);

    // 显示窗口和所有子控件
    gtk_widget_show_all(window);

    // 进入主事件循环
    gtk_main();

}

#else //HAVE_GTK
int abcdk_test_gtk(abcdk_option_t *args)
{
    return 0;
}
#endif //HAVE_GTK
