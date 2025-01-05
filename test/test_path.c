/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

int abcdk_test_path(abcdk_option_t *args)
{
    char buf[PATH_MAX] ={0};
    char test[] = {"sfsdfas/fdasdf/././../ewter/../../ewrtertert/wertwertert/wertwetwert./../..//e/e/e//t/t/y/y//f/f/f/f/f/ff/"};
   // char test[] = {"aaAA/bbbb.jpg"};

    abcdk_abspath(test,0);
    fprintf(stderr,"%s\n",test);

    int chk = abcdk_fnmatch("test_record/新建文件夹 (2)/test_ffmpeg/x64/Debug/test_ffmpeg.dll","*/*/*/*/*/*.dll",1,1);
    printf("chk=%d\n",chk);

    chk = abcdk_fnmatch("test_record/新建文件夹 (2)/test_ffmpeg/x64/Debug/test_ffmpeg.dll","*.dll",0,0);
    printf("chk=%d\n",chk);

    return 0;
}
