/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/shell/file.h"

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

uint64_t _abcdk_file_segment_find_pos(const char *dst, uint64_t start)
{
    abcdk_tree_t *dir = NULL;
    char tmp[PATH_MAX] = {0},tmp2[NAME_MAX] = {0},tmp3[NAME_MAX] = {0};
    uint64_t pos = 0,pos2 = 0;
    int chk;

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

        chk = sscanf(tmp3, tmp2, &pos2);
        if (chk != 1)
            continue;

        /*记录最大编号。*/
        if (pos < pos2)
            pos = pos2;
    }

no_history_file:

    /*可能不存在。*/
    if (pos < start)
        pos = start;

    abcdk_tree_free(&dir);

    return pos;
}

int abcdk_file_segment(const char *src, const char *dst, uint64_t start,uint16_t count,uint64_t *prev2next)
{
    abcdk_tree_t *dir = NULL;
    char tmp[PATH_MAX] = {0};
    char tmp2[PATH_MAX] = {0};
    uint64_t pos;
    int chk;

    assert(src != NULL && dst != NULL && start > 0 && count > 0 && prev2next != NULL);

    /*copy*/
    pos = *prev2next;

    /*如果未输入编号，则需要主动查找现存的最大编号。*/
    if(pos < start)
        pos = _abcdk_file_segment_find_pos(dst,start);


    /*如果超过保留数量，则删除编号最小的文件。*/
    if(count <= pos - start)
    {
        snprintf(tmp, PATH_MAX, dst, pos - count);
        chk = remove(tmp);
        if (chk != 0  && errno != ENOENT)
            return -1;
    }

    snprintf(tmp2, PATH_MAX, dst, pos);
    chk = rename(src,tmp2);
    if (chk != 0)
        return -1;

    /*next POS。*/
    *prev2next = pos +1;

    return 0;
}