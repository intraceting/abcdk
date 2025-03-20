/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/system/dmi.h"

int _abcdk_dmi_hash_dump_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    /*已经结束。*/
    if(depth == SIZE_MAX)
        return -1;

    if(depth == 0)
        abcdk_tree_fprintf(stderr,depth,node,"%s\n",__FUNCTION__);
    else 
        abcdk_tree_fprintf(stderr,depth,node,"%s\n",node->obj->pstrs[0]);

    return 1;
}

int _abcdk_dmi_hash_compare_cb(const abcdk_tree_t *node1, const abcdk_tree_t *node2, void *opaque)
{
    return abcdk_strcmp(node1->obj->pstrs[0],node2->obj->pstrs[0],1);
}

static int _abcdk_dmi_check_part_removable(char *part_name)
{
    abcdk_tree_t *dir_ctx = NULL;
    char dev_file[PATH_MAX];
    char part_file[PATH_MAX];
    char dev_name[NAME_MAX];
    uint8_t removable_flag;
    int chk;

    chk = abcdk_dirent_open(&dir_ctx, "/sys/block/");
    if (chk != 0)
        return -1; // 打开失败。

    while (1)
    {
        memset(dev_file, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir_ctx, NULL, dev_file, 1);
        if (chk != 0)
            break;

        /*把分区名字作为当前路径的的子目录拼接。*/

        memset(part_file, 0, PATH_MAX);
        abcdk_dirdir(part_file, dev_file);
        abcdk_dirdir(part_file, part_name);

        /*检查文件(分区)是否存在。*/
        chk = access(part_file, F_OK);
        if (chk == 0)
            break;

        /*走到这里表示设备没有分区。*/

        /*提取设备名称。*/
        memset(dev_name, 0, NAME_MAX);
        abcdk_basename(dev_name, dev_file);

        /*检查分区名称是否就是设备名称。*/
        chk = abcdk_strcmp(dev_name, part_name, 1);
        if (chk == 0)
            break;
    }

    abcdk_tree_free(&dir_ctx);//free.

    if (dev_file[0] == '0')
        return -1; // 未找到。

    /*检查removable文是否可读。*/
    abcdk_dirdir(dev_file, "removable");
    chk = access(dev_file, R_OK);
    if (chk != 0)
        return -1; // 未知的。

    chk = abcdk_load(dev_file, &removable_flag, 1, 0);
    if (chk != 1)
        return -1; // 未知的。

    if (removable_flag != '0')
        return 1; // 可移动设备。

    return 0; // 固定设备。
}

int abcdk_dmi_hash(uint8_t uuid[16], const char *stuff)
{
    abcdk_md5_t *md5_ctx = NULL;
    abcdk_tree_t *dir_ctx = NULL;
    abcdk_tree_t *keyword_list = NULL,*p = NULL;
    char tmp_file[PATH_MAX];
    char tmp_path[PATH_MAX];
    char part_file[PATH_MAX];
    char part_link[PATH_MAX];
    char part_name[NAME_MAX];
    char alias_name[NAME_MAX];
    int chk;

    keyword_list = abcdk_tree_alloc3(1);
    if(!keyword_list)
        return -1;

    chk = abcdk_dirent_open(&dir_ctx,"/dev/disk/by-uuid");
    if(chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING,"打开目录(%s)失败，无权限或不存在，忽略。","/dev/disk/by-uuid");
    }

    chk = abcdk_dirent_open(&dir_ctx,"/dev/disk/by-partuuid");
    if(chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING,"打开目录(%s)失败，无权限或不存在，忽略。","/dev/disk/by-partuuid");
    }

    chk = abcdk_dirent_open(&dir_ctx,"/dev/disk/by-id");
    if(chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING,"打开目录(%s)失败，无权限或不存在，忽略。","/dev/disk/by-id");
    }

    while(1)
    {
        memset(tmp_file,0,PATH_MAX);
        chk = abcdk_dirent_read(dir_ctx,NULL,tmp_file,1);
        if(chk != 0)
            break;
        
        memset(part_file,0,PATH_MAX);
        memset(part_link,0,PATH_MAX);
        
        /*读取软链接路径,。*/
        chk = readlink(tmp_file, part_link, PATH_MAX);
        if (chk <= 0)
            continue;
        
        /*提取目录。*/
        abcdk_dirname(part_file,tmp_file);

        /*拼接软链接路径。*/
        abcdk_dirdir(part_file,part_link);

        /*去掉冗余的路径。*/
        abcdk_abspath(part_file,0);

        /*验证是否存在/dev/XXXX。*/
        chk = access(part_file, F_OK);
        if (chk != 0)
            continue;

        /*提取分区名字。*/
        memset(part_name,0,NAME_MAX);        
        abcdk_basename(part_name,part_file);

        /*检查分区所属的设备是否可移动。*/
        chk = _abcdk_dmi_check_part_removable(part_name);
        if(chk != 0)
            continue;

        /*提取分区别名。*/
        abcdk_basename(alias_name,tmp_file);

        abcdk_trace_printf(LOG_DEBUG,"块设备分区的别名(%s)将被用于计算DMI的哈希值。",alias_name);

        /*创建关键字节点，并复制分区别名。*/
        p = abcdk_tree_alloc4(alias_name, strlen(alias_name));
        if (!p)
            continue;

        /*追加到关键列表。*/
        abcdk_tree_insert2(keyword_list, p, 0);
    }

    abcdk_tree_free(&dir_ctx);//free.

    /*检查列表里是否有数据。*/
    p = abcdk_tree_child(keyword_list, 1);//first.
    if(!p)
    {
        abcdk_trace_printf(LOG_ERR,"没有发现可用于计算DMI哈希值的固定块设备");

        abcdk_tree_free(&keyword_list);//free.
        return -1;
    }

    /*如果填充物有效，追加到列表中。*/
    if(stuff && *stuff)
    {
        p = abcdk_tree_alloc4(stuff, strlen(stuff));
        if (!p)
        {
            abcdk_tree_free(&keyword_list);//free.
            return -1;
        }

        abcdk_tree_insert2(keyword_list, p, 0);
    }

    abcdk_tree_iterator_t it = {0,NULL,_abcdk_dmi_hash_dump_cb,_abcdk_dmi_hash_compare_cb};

    /*排序并去重。*/
    abcdk_tree_sort(keyword_list,&it,1);
    abcdk_tree_distinct(keyword_list,&it);

    /*打印。*/
    //abcdk_tree_scan(keyword_list,&it);

    /*创建MD5环境。*/
    md5_ctx = abcdk_md5_create();
    if(!md5_ctx)
    {
        abcdk_tree_free(&keyword_list);
        return -1;
    }

    /*遍历表表。*/
    p = abcdk_tree_child(keyword_list, 1);//first.
    while (p)
    {
        abcdk_md5_update(md5_ctx,p->obj->pstrs[0],strlen(p->obj->pstrs[0]));

        p = abcdk_tree_sibling(p, 0);//next.
    }
    
    abcdk_md5_final(md5_ctx,uuid);
    abcdk_md5_destroy(&md5_ctx);

    abcdk_tree_free(&keyword_list);//free.

    return 0;
}