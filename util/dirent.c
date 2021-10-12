/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/dirent.h"

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
        return -1;

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
        abcdk_tree_unlink(tmp);
        abcdk_tree_free(&tmp);
        goto prev;
    }

    if (abcdk_strcmp(c_dir->d_name, ".", 1) == 0 || abcdk_strcmp(c_dir->d_name, "..", 1) == 0)
        goto next;

    abcdk_dirdir(file, (char*)tmp->alloc->pptrs[0]);
    abcdk_dirdir(file, c_dir->d_name);

    return 0;    
}