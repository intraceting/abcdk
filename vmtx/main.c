/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "context.h"
#include "service.h"


int main(int argc, char **argv)
{
    abcdk_tree_t *args = NULL;
    int errcode = 0;

    /*中文；UTF-8。*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    /*随机数种子。*/
    srand(time(NULL));

    /*记录日志。*/
    abcdk_openlog(NULL, LOG_INFO, 1);

    /*申请参数存储空间。*/
    args = abcdk_tree_alloc3(1);
    if (!args)
        ABCDK_ERRNO_AND_GOTO1(errcode = errno,final);
    
    /*解析参数。*/
    abcdk_getargs(args, argc, argv, "--");

final:
    
    abcdk_tree_free(&args);

    return errcode;
}