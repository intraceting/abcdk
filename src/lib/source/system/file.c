/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/system/file.h"

int abcdk_file_wholockme(const char *file,int pids[],int max)
{
    char cmd[PATH_MAX] = {0};
    pid_t t = -1;
    int ofd = -1;
    FILE *rfd = NULL;
    int fc[2] = {0};
    char *line_p = NULL;
    size_t line_l = 0;
    ssize_t rlen = 0;
    int line_c = 0;
    int status = 0;
    int exitcode = 0;

    assert(file != NULL && pids != NULL && max > 0);

    snprintf(cmd,PATH_MAX,"lsof -F pf \"%s\"",file);

    /*如果无法执行查询，则返回无进程占用文件。*/
    t = abcdk_popen(cmd,NULL,0,0, NULL, NULL, NULL, &ofd, NULL);
    if (t < 0)
        return -1;

    rfd = fdopen(ofd,"r");
    if(!rfd)
        return -2;

    while (1)
    {
        rlen = abcdk_fgetline(rfd, &line_p, &line_l, '\n', 0);
        if(rlen < 0)
            break;
            
        /*跳过无法存储的（不能超过数组容量），不然管道中有数据但无“人”读取的话，会卡死父进程。*/
        if (line_c >= max)
            continue;

        if (*line_p == 'p')
        {
            pids[line_c] = strtol(line_p + 1, NULL, 0);
            fc[0] += 1;
        }
        else if (*line_p == 'f')
        {
            fc[1] += 1;
        }

        /*每行的所有字段都读取后，行号滚动。*/
        if (fc[0] == fc[1])
            line_c += 1;
    }

    /*获取子进程退出状态，防止出现僵尸子进程。*/
    waitpid(t,&status,0);
    exitcode = WIFEXITED(status);

    if(rfd)
        fclose(rfd);
    if(line_p)
        free(line_p);

    return line_c;
}

void _abcdk_file_segment_find_pos(const char *dst, uint64_t start, uint64_t pos[2])
{
    abcdk_tree_t *dir = NULL;
    char tmp[PATH_MAX] = {0},tmp2[PATH_MAX] = {0},tmp3[PATH_MAX] = {0};
    uint64_t pos_tmp = 0;
    int chk;
    
    pos[0] = UINT64_MAX;
    pos[1] = 0;

    abcdk_dirname(tmp, dst);
    abcdk_basename(tmp2, dst);

    chk = abcdk_dirent_open(&dir, tmp);
    if (chk != 0)
        goto no_history_file;

    while (1)
    {
        memset(tmp3, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir, NULL, tmp3, 0);
        if (chk != 0)
            break;

        chk = sscanf(tmp3, tmp2, &pos_tmp);
        if (chk != 1)
            continue;

        if (start > pos_tmp)
            continue;

        if(pos[0] > pos_tmp)
            pos[0] = pos_tmp;

        if(pos[1] < pos_tmp)
            pos[1] = pos_tmp;
    }

no_history_file:

    if(pos[1] == 0)
        pos[1] = start;

    if (pos[0] > pos[1])
        pos[0] = pos[1];

    abcdk_tree_free(&dir);

    return;
}

int abcdk_file_segment(const char *src,const char *dst, uint16_t winsize, uint64_t start, uint64_t pos[2])
{
    char tmp[PATH_MAX] = {0};
    uint64_t pos_min= 0,pos_max = 0;
    int chk;

    assert(dst != NULL && winsize > 0 && start > 0 && pos != NULL);

    /*重启生产。*/
    if (pos[0] > pos[1])
        _abcdk_file_segment_find_pos(dst,start,pos);

    /*有序的增长最大编号。*/
    for (; pos[1] < UINT64_MAX; pos[1]++)
    {
        memset(tmp,0,PATH_MAX);
        snprintf(tmp, PATH_MAX, dst, pos[1]);
        chk = access(tmp, F_OK);
        if (chk != 0)
            break;
    }

    if(src)
    {
        memset(tmp, 0, PATH_MAX);
        snprintf(tmp, PATH_MAX, dst, pos[1]);
        chk = rename(src, tmp);
        if (chk != 0)
            return -2;
    }

    /*删除冗余窗口之前的。*/
    for (; pos[0] < pos[1]; pos[0]++)
    {
        if (pos[1] - pos[0] < winsize)
            break;

        memset(tmp, 0, PATH_MAX);
        snprintf(tmp, PATH_MAX, dst, pos[0]);
        chk = remove(tmp);
        if (chk != 0 && errno != ENOENT)
            return -1;
    }


    return 0;
}

int abcdk_file_segment_append(const char *src, const char *dst, uint64_t pos)
{
    char tmp[PATH_MAX] = {0};
    int chk;

    assert(src != NULL && dst != NULL && pos > 0);
    
    memset(tmp,0,PATH_MAX);
    snprintf(tmp, PATH_MAX, dst, pos);
    chk = rename(src, tmp);
    if (chk != 0)
        return -1;

    return 0;
}
