/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include "abcdk/shell/mtab.h"

const char *_abcdk_mtab_split_field(char **pos)
{
    char *b = NULL,*p = NULL;

    /*找到字段开头。*/
    p = *pos;
    while (*p != '\0' && isspace(*p))
    {
        p++;
    }

    /*记录字段头指针。*/
    b = p;

    /*找到字段末尾。*/
    while (*p != '\0' && !isspace(*p))
    {
        p++;
    }

    /*修改末尾结束符。*/
    *p = '\0';

    /*指针向后移动并保存。*/
    *pos = p + 1;

    return b;
}

void abcdk_mtab_list(abcdk_tree_t *list)
{
    abcdk_tree_t *dev = NULL;
    abcdk_mtab_info_t *dev_p = NULL;
    FILE *fp = NULL;
    char *line = NULL;
    size_t len = 0;
    ssize_t rlen = 0;
    char *pos = NULL;

    assert(list != NULL);

    fp = fopen("/proc/self/mounts", "r");
    if (!fp)
        return;

    while (1)
    {
        rlen = abcdk_fgetline(fp, &line, &len, '\n', 0);
        if(rlen < 0)
            break;

        size_t sizes[] = {sizeof(abcdk_mtab_info_t), rlen + 1};
        dev = abcdk_tree_alloc2(sizes, 2, 0);
        if (!dev)
            break;

        abcdk_tree_insert2(list,dev,0);

        strncpy((char*)dev->obj->pptrs[1],line,rlen);
        
        dev_p = (abcdk_mtab_info_t*)dev->obj->pptrs[0];
        pos = (char*)dev->obj->pptrs[1];

        /*字段分割，顺序不能换。*/
        dev_p->fs = _abcdk_mtab_split_field(&pos);
        dev_p->mpoint = _abcdk_mtab_split_field(&pos);
        dev_p->type = _abcdk_mtab_split_field(&pos);
        dev_p->options = _abcdk_mtab_split_field(&pos);
        dev_p->dump = _abcdk_mtab_split_field(&pos);
        dev_p->pass = _abcdk_mtab_split_field(&pos);
    }

final:

    /*不要忘记释放这块内存，不然可能会有内存泄漏的风险。 */
    if (line)
        free(line);
    if (fp)
        fclose(fp);
}