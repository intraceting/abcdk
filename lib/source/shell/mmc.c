/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/shell/mmc.h"

int _abcdk_mmc_get_type(const char *path, char type[NAME_MAX])
{
    ssize_t rlen = 0;
    char file[PATH_MAX] = {0};

    abcdk_dirdir(file,path);
    abcdk_dirdir(file,"type");

    rlen = abcdk_load(file, type, NAME_MAX-1, 0);
    if (rlen <= 0)
        return -1;

    abcdk_strtrim(type,isspace,2);
    
    return 0;
}

int _abcdk_mmc_get_cid(const char *path, char cid[NAME_MAX])
{
    ssize_t rlen = 0;
    char file[PATH_MAX] = {0};

    abcdk_dirdir(file,path);
    abcdk_dirdir(file,"cid");

    rlen = abcdk_load(file, cid, NAME_MAX-1, 0);
    if (rlen <= 0)
        return -1;

    abcdk_strtrim(cid,isspace,2);
    
    return 0;
}

int _abcdk_mmc_get_name(const char *path, char name[NAME_MAX])
{
    ssize_t rlen = 0;
    char file[PATH_MAX] = {0};

    abcdk_dirdir(file,path);
    abcdk_dirdir(file,"name");

    rlen = abcdk_load(file, name, NAME_MAX-1, 0);
    if (rlen <= 0)
        return -1;

    abcdk_strtrim(name,isspace,2);
    
    return 0;
}

int _abcdk_mmc_get_devname(const char *path,const char *type, char devname[NAME_MAX])
{
    char buf[PATH_MAX] = {0};
    char buf2[PATH_MAX] = {0};
    char buf3[40] = {0};
    int deflag = 0,major = -1,minor = -1;
    abcdk_tree_t *dir = NULL;
    char path2[PATH_MAX] = {0};
    int chk;

    abcdk_dirdir(path2, path);
    if(abcdk_strcmp(type,"SD",0) == 0)
        abcdk_dirdir(path2, "block");
    else if(abcdk_strcmp(type,"MMC",0) == 0)
        abcdk_dirdir(path2, "block");
    else
        return -1;
    
    dir = abcdk_tree_alloc3(1);
    if (!dir)
        return -1;

    chk = abcdk_dirent_open(&dir, path2);
    if (chk != 0)
        goto final;

    while (1)
    {
        memset(buf, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir,NULL, buf,1);
        if (chk != 0)
            break;

        /*暂存设备名字。*/
        memset(devname, 0, NAME_MAX);
        abcdk_dirdir(devname, "/dev/");
        abcdk_basename(devname + strlen(devname), buf);

        /*检查/dev/下是否存在。*/
        if (access(devname, F_OK) != 0)
            continue;

        /*存在则跳出。*/
        break;
    }

    /*No error.*/
    chk = 0;

final:

    abcdk_tree_free(&dir);

    return chk;
}

int abcdk_mmc_get_info(const char *path,abcdk_mmc_info_t *info)
{
    int chk;

    assert(path != NULL && info != NULL);

    chk = _abcdk_mmc_get_type(path, info->type);
    if (chk != 0)
        return -1;

    chk = _abcdk_mmc_get_cid(path,info->cid);
    if (chk != 0)
        return -1;

    chk = _abcdk_mmc_get_name(path,info->name);
    if (chk != 0)
        return -1;

    _abcdk_mmc_get_devname(path,info->type,info->devname);

    return 0;
}

void abcdk_mmc_list(abcdk_tree_t *list)
{
    abcdk_tree_t *dev = NULL;
    abcdk_mmc_info_t *dev_p = NULL;
    abcdk_tree_t *dir = NULL;
    char path[PATH_MAX] = {0};
    char type[NAME_MAX];
    int chk;

    assert(list != NULL);

    dir = abcdk_tree_alloc3(1);
    if (!dir)
        return;

    chk = abcdk_dirent_open(&dir, "/sys/bus/mmc/devices");
    if (chk != 0)
        goto final;

    /*遍历目录。*/
    while (1)
    {
        memset(path, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir,NULL, path,1);
        if (chk != 0)
            break;

        /*获取设备类型(可能会失败)。*/
        memset(type, 0, NAME_MAX);
        chk = _abcdk_mmc_get_type(path,type);

        /*跳过无法获取类型的设备。*/
        if (chk != 0)
            continue;

        dev = abcdk_tree_alloc3(sizeof(abcdk_mmc_info_t));
        if(!dev)
            break;

        dev_p = (abcdk_mmc_info_t*)dev->obj->pptrs[0];
        abcdk_tree_insert2(list,dev,0);
        
        /*从路径中分离bus并保存。*/
        abcdk_basename(dev_p->bus,path);
        /*提取其它字段并保存。*/
        abcdk_mmc_get_info(path,dev_p);
        
    }

final:

    abcdk_tree_free(&dir);

    return;
}


int _abcdk_mmc_find(abcdk_tree_t *list, abcdk_tree_t *node)
{
    abcdk_tree_t *p;
    abcdk_mmc_info_t *dev_p, *dev_q;

    /*链表为空，直接返回“未找到”。*/
    if (!list)
        return 0;

    dev_q = (abcdk_mmc_info_t *)node->obj->pptrs[0];

    p = abcdk_tree_child(list, 1);
    while (p)
    {
        dev_p = (abcdk_mmc_info_t *)p->obj->pptrs[0];

        if (abcdk_strcmp(dev_p->bus, dev_q->bus,1) == 0 &&
            abcdk_strcmp(dev_p->devname, dev_q->devname,1) == 0)
        {
            return 1;
        }

        p = abcdk_tree_sibling(p, 0);
    }

    return 0;
}

void _abcdk_mmc_diff(abcdk_tree_t *old_list,abcdk_tree_t *new_list,abcdk_tree_t **diff,int add)
{
    abcdk_tree_t *p;
    abcdk_tree_t *tmp;
    abcdk_tree_t *diff_p;

    abcdk_tree_free(diff);
    *diff = abcdk_tree_alloc3(1);
    if (!*diff)
        return;

    diff_p = *diff;

    /*
     * add == 1 : 从旧的中查找新的。
     * del == 0 ; 从新的中查找旧的。
    */
   
    p = abcdk_tree_child((add ? new_list : old_list), 1);
    while (p)
    {
        if (!_abcdk_mmc_find((add ? old_list : new_list), p))
        {
            tmp = abcdk_tree_alloc(abcdk_object_refer(p->obj));//增加引用计数。
            if (!tmp)
                return;

            abcdk_tree_insert2(diff_p, tmp, 0);
        }

        p = abcdk_tree_sibling(p, 0);
    }

    /*如果未发生变化，删除差异链表。*/
    if(!abcdk_tree_child(diff_p,1))
        abcdk_tree_free(diff);

    return;
}

void _abcdk_mmc_check_ok(abcdk_tree_t *list)
{
    abcdk_tree_t *p,*tmp;
    abcdk_mmc_info_t *dev_p;

    p = abcdk_tree_child(list, 1);
    while (p)
    {
        dev_p = (abcdk_mmc_info_t *)p->obj->pptrs[0];

        if (dev_p->devname[0] == '\0')
        {
            /*下一个节点。*/
            p = abcdk_tree_sibling(tmp = p, 0);

            /*删除不完成的节点。*/
            abcdk_tree_unlink(tmp);
            abcdk_tree_free(&tmp);
        }
        else 
        {
            /*下一个节点。*/
            p = abcdk_tree_sibling(p, 0);
        }
    }
}

void abcdk_mmc_watch(abcdk_tree_t **snapshot, abcdk_tree_t **add, abcdk_tree_t **del)
{
    abcdk_tree_t *tmp = NULL;

    assert(snapshot != NULL);

    tmp = abcdk_tree_alloc3(1);
    if (!tmp)
        return;

    abcdk_mmc_list(tmp);
    _abcdk_mmc_check_ok(tmp);

    if(*snapshot)
    {
        if(add)
            _abcdk_mmc_diff(*snapshot,tmp,add,1);
        if(del)
            _abcdk_mmc_diff(*snapshot,tmp,del,0);
    }
    else
    {
        if(add)
            _abcdk_mmc_diff(NULL,tmp,add,1);
    }
    
    abcdk_tree_free(snapshot);
    _abcdk_mmc_diff(NULL,tmp,snapshot,1);
    
    abcdk_tree_free(&tmp);
}
