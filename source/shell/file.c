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

int abcdk_file_segment(const char *src, const char *dst, int max)
{
    char tmp[PATH_MAX] = {0};
    char tmp2[PATH_MAX] = {0};
    int chk;

    assert(src != NULL && dst != NULL && max > 0);

    /*依次修改分段文件编号。*/
    for (int i = max; i > 0; i--)
    {
        /*编号较大的分段文件。*/
        snprintf(tmp2, PATH_MAX, dst, i);

        /*删除编号最大的分段文件。*/
        if (i == max)
        {
            if (access(tmp2, F_OK) == 0)
            {
                chk = remove(tmp2);
                if (chk != 0)
                    return -1;
            }
        }

        /*编号较小的分段文件。*/
        if (i > 1)
            snprintf(tmp, PATH_MAX, dst, i - 1);
        else
            strncpy(tmp, src, PATH_MAX);

        /*跳过不存在的分段文件。*/
        if (access(tmp, F_OK) != 0)
            continue;

        chk = rename(tmp,tmp2);
        if (chk != 0)
            return -1;
    }

    return 0;
}