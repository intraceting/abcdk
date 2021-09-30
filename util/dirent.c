/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/dirent.h"

void _abcdk_dirent_scan(abcdk_tree_t *father, size_t depth, abcdk_dirent_filter_t *filter)
{
    DIR *f_dir = NULL;
    struct dirent *c_dir = NULL;
    char *f_path = NULL;
    struct stat *f_stat = NULL;
    char *c_path = NULL;
    struct stat *c_stat = NULL;
    abcdk_tree_t *node = NULL;
    abcdk_tree_t *tmp = NULL;

    abcdk_dirent_counter_t *counter_p = NULL;
    int chk;

    f_path = (char *)(father->alloc->pptrs[ABCDK_DIRENT_NAME]);
    f_stat = (struct stat *)(father->alloc->pptrs[ABCDK_DIRENT_STAT]);

    assert(f_path != NULL && *f_path != '\0');
    assert(f_stat != NULL);

    /* 可能是递归进来的，查询一次即可。*/
    if (f_stat->st_nlink <= 0)
    {
        if (lstat(f_path, f_stat) == -1)
            return;
    }

    if (!S_ISDIR(f_stat->st_mode))
        ABCDK_ERRNO_AND_RETURN0(ENOTDIR);

    f_dir = opendir(f_path);
    if (!f_dir)
        return;

    for(;;)
    {
        c_dir = readdir(f_dir);
        if(!c_dir)
            break;

        if (abcdk_strcmp(c_dir->d_name, ".", 1) == 0 || abcdk_strcmp(c_dir->d_name, "..", 1) == 0)
            continue;

        size_t sizes[9] = {PATH_MAX, sizeof(struct stat), sizeof(abcdk_dirent_counter_t),
                           sizeof(abcdk_dirent_counter_t), sizeof(abcdk_dirent_counter_t),
                           sizeof(abcdk_dirent_counter_t), sizeof(abcdk_dirent_counter_t),
                           sizeof(abcdk_dirent_counter_t),sizeof(abcdk_dirent_counter_t)};

        node = abcdk_tree_alloc2(sizes, ABCDK_ARRAY_SIZE(sizes), 0);
        if (!node)
            break;

        c_path = (char *)(node->alloc->pptrs[ABCDK_DIRENT_NAME]);
        c_stat = (struct stat *)(node->alloc->pptrs[ABCDK_DIRENT_STAT]);

        abcdk_dirdir(c_path, f_path);
        abcdk_dirdir(c_path, c_dir->d_name);

        /*node->d_type 在某些文件系统中并未正确填写有效值，因此不能直接使用，这里用替待方案。 */
        if (lstat(c_path, c_stat) == -1)
        {
            abcdk_tree_free(&node);
            continue;
        }

        chk = 0;
        if(filter && filter->match_cb)
            chk = filter->match_cb(depth,node,filter->opaque);

        /* 是否跳过。*/
        if (chk == -1)
        {
            abcdk_tree_free(&node);
            continue;
        }

        /* 是否终止。*/
        if (chk == -2)
        {
            abcdk_tree_free(&node);
            break;
        }

        /*加入到树节点。*/
        abcdk_tree_insert2(father, node, 0);

        /* 统计。*/
        tmp = node;
        for(;;)
        {
            tmp = abcdk_tree_father(tmp);
            if(!tmp)
                break;

            if (S_ISDIR(c_stat->st_mode))
                counter_p = ABCDK_PTR2PTR(abcdk_dirent_counter_t,tmp->alloc->pptrs[ABCDK_DIRENT_DIRS], 0);
            else if (S_ISCHR(c_stat->st_mode))
                counter_p = ABCDK_PTR2PTR(abcdk_dirent_counter_t,tmp->alloc->pptrs[ABCDK_DIRENT_CHRS], 0);
            else if (S_ISBLK(c_stat->st_mode))
                counter_p = ABCDK_PTR2PTR(abcdk_dirent_counter_t,tmp->alloc->pptrs[ABCDK_DIRENT_BLKS], 0);
            else if (S_ISREG(c_stat->st_mode))
                counter_p = ABCDK_PTR2PTR(abcdk_dirent_counter_t,tmp->alloc->pptrs[ABCDK_DIRENT_REGS], 0);
            else if (S_ISFIFO(c_stat->st_mode))
                counter_p = ABCDK_PTR2PTR(abcdk_dirent_counter_t,tmp->alloc->pptrs[ABCDK_DIRENT_FIFOS], 0);
            else if (S_ISLNK(c_stat->st_mode))
                counter_p = ABCDK_PTR2PTR(abcdk_dirent_counter_t,tmp->alloc->pptrs[ABCDK_DIRENT_LNKS], 0);
            else if (S_ISSOCK(c_stat->st_mode))
                counter_p = ABCDK_PTR2PTR(abcdk_dirent_counter_t,tmp->alloc->pptrs[ABCDK_DIRENT_SOCKS], 0);
            else 
                counter_p = NULL;

            if(counter_p)
            {
                counter_p->nums += 1;
                counter_p->sizes += (S_ISREG(c_stat->st_mode)?c_stat->st_size:0);
            }
            
        }

        /* 递归。 */
        if (chk == 0)
            _abcdk_dirent_scan(node, depth + 1, filter);
    }

    if (f_dir)
        closedir(f_dir);
}

abcdk_tree_t *abcdk_dirent_scan(const char *path, abcdk_dirent_filter_t *filter)
{
    abcdk_tree_t *root = NULL;

    assert(path != NULL);

    if (access(path, R_OK) != 0)
        return NULL;

    size_t sizes[9] = {PATH_MAX, sizeof(struct stat), sizeof(abcdk_dirent_counter_t),
                       sizeof(abcdk_dirent_counter_t), sizeof(abcdk_dirent_counter_t),
                       sizeof(abcdk_dirent_counter_t), sizeof(abcdk_dirent_counter_t),
                       sizeof(abcdk_dirent_counter_t),sizeof(abcdk_dirent_counter_t)};

    root = abcdk_tree_alloc2(sizes, ABCDK_ARRAY_SIZE(sizes), 0);
    if (!root)
        return NULL;

    strcpy(root->alloc->pptrs[ABCDK_DIRENT_NAME], path);

    _abcdk_dirent_scan(root, 0, filter);

    return root;
}

void _abcdk_dirent_destroy_cb(abcdk_allocator_t *alloc, void *opaque)
{
    if (alloc->pptrs[1])
        closedir((DIR *)alloc->pptrs[1]);
}

int abcdk_dirent_open(abcdk_tree_t *dir,const char *path)
{
    abcdk_tree_t *tmp = NULL;

    assert(dir != NULL && path != NULL);

    if (access(path, R_OK) != 0)
        return NULL;

    size_t sizes[2] = {PATH_MAX,0};
    tmp = abcdk_tree_alloc2(sizes,2,0);
    if(!tmp)
        return -1;

    abcdk_allocator_atfree(tmp->alloc,_abcdk_dirent_destroy_cb,NULL);

    tmp->alloc->pptrs[1] = (uint8_t*)opendir(path);
    if (!tmp->alloc->pptrs[1])
        ABCDK_ERRNO_AND_GOTO1(errno,final_error);

    strncpy(tmp->alloc->pptrs[0],path,PATH_MAX);

    abcdk_tree_insert2(dir,tmp,0);

    return 0;

final_error:

    abcdk_tree_free(&tmp);

    return -1;
}

int abcdk_dirent_read(abcdk_tree_t *dir,char file[PATH_MAX])
{
    abcdk_tree_t *tmp = NULL;
    struct dirent *c_dir = NULL;

    assert(dir != NULL && file != NULL);

prev:

    tmp = abcdk_tree_child(dir,0);
    if(!tmp)
        return -1;

next:

    c_dir = readdir((DIR*)tmp->alloc->pptrs[1]);
    if(!c_dir)
    {
        abcdk_tree_free(&tmp);
        goto prev;
    }

    if (abcdk_strcmp(c_dir->d_name, ".", 1) == 0 || abcdk_strcmp(c_dir->d_name, "..", 1) == 0)
        goto next;

    abcdk_dirdir(file, (char*)tmp->alloc->pptrs[0]);
    abcdk_dirdir(file, c_dir->d_name);

    return 0;    
}