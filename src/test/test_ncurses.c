/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "entry.h"

#ifdef HAVE_NCURSES
#include <ncurses.h>

int abcdk_test_ncurses(abcdk_option_t *args)
{

 // 初始化 ncurses
    initscr();              
    noecho();               
    curs_set(0);            // 隐藏光标

    // 获取终端尺寸
    int screen_height, screen_width;
    getmaxyx(stdscr, screen_height, screen_width);

    // 窗体尺寸和位置
    int win_height = 10;
    int win_width = 40;
    int start_y = (screen_height - win_height) / 2;
    int start_x = (screen_width - win_width) / 2;

    // 创建窗体
    WINDOW *win = newwin(win_height, win_width, start_y, start_x);
    box(win, 0, 0);         // 添加边框

    // 要显示的两行文字
    const char *line1 = "This is the first line.";
    const char *line2 = "This is the second line.";

    // 计算文字位置并打印到窗体
    int text_start_y = 2;   // 行位置(窗体内部的第几行)
    int text_start_x = 2;   // 列位置(左对齐距离边框2个字符)

    mvwprintw(win, text_start_y, text_start_x, "%s", line1);
    mvwprintw(win, text_start_y + 2, text_start_x, "%s", line2);

    // 显示窗体
    wrefresh(win);

    // 等待用户输入
    getch();

    // 清理并退出
    delwin(win);            // 删除窗体
    endwin();  

    return 0;
}

#else //HAVE_NCURSES
int abcdk_test_ncurses(abcdk_option_t *args)
{
    return 0;
}
#endif //HAVE_NCURSES
