/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/shell/block.h"

int abcdk_block_find_device(const char *name, char devpath[PATH_MAX])
{
    abcdk_tree_t *dir = NULL;
    char path[PATH_MAX] = {0};
    char path2[PATH_MAX] = {0};
    char name2[NAME_MAX] = {0};
    int chk;

    assert(name != NULL && path != NULL);

    dir = abcdk_tree_alloc3(1);
    if (!dir)
        return -1;

    chk = abcdk_dirent_open(&dir, "/sys/block/");
    if (chk != 0)
        goto final;

    /*遍历目录。*/
    while (1)
    {
        memset(path, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir,NULL, path,1);
        if (chk != 0)
            break;

        /*截取名字。*/
        abcdk_basename(name2, path);

        /*比较当前名字。*/
        if (abcdk_strcmp(name, name2, 1) != 0)
        {
            /*拼接子级名字。*/
            memset(path2, 0, PATH_MAX);
            abcdk_dirdir(path2, path);
            abcdk_dirdir(path2, name);

            /*比较子级名字。*/
            if (access(path2, F_OK) != 0)
                continue;
        }

        /*拼接成设备信息路径。*/
        abcdk_dirdir(devpath, path);
        abcdk_dirdir(devpath, "device");

        /*走到这里，表示找到了。*/
        chk = 0;
        break;
    }

final:

    abcdk_tree_free(&dir);

    return chk;
}