/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/path.h"

char *abcdk_dirdir(char *path, const char *suffix)
{
    size_t len = 0;

    assert(path != NULL && suffix != NULL);

    len = strlen(path);
    if (len > 0)
    {
        if (path[len - 1] == '/')
        {
            if (suffix[0] == '/')
                suffix += 1;
        }
        else if (suffix[0] != '/') 
        {
            strcat(path, "/");
        }
    }

    strcat(path, suffix);

    return path;
}

void abcdk_mkdir(const char *path, mode_t mode)
{
    size_t len = 0;
    char *tmp = NULL;
    int chk = 0;

    assert(path != NULL);

    len = strlen(path);
    if (len <= 0)
        ABCDK_ERRNO_AND_RETURN0(EINVAL);

    tmp = (char *)abcdk_heap_clone(path, len);
    if (!tmp)
        ABCDK_ERRNO_AND_RETURN0(ENOMEM);

    /* 必须允许当前用户具有读、写、执行权限。 */
    mode |= S_IRWXU;

    for (size_t i = 1; i < len; i++)
    {
        if (tmp[i] != '/')
            continue;

        tmp[i] = '\0';

        if (access(tmp, F_OK) != 0)
            chk = mkdir(tmp, mode & (S_IRWXU | S_IRWXG | S_IRWXO));

        tmp[i] = '/';

        if (chk != 0)
            break;
    }

    if (tmp)
        abcdk_heap_freep((void**)&tmp);
}

char *abcdk_dirname(char *dst, const char *src)
{
    char *find = NULL;
    char *path = NULL;

    assert(dst != NULL && src != NULL);

    path = (char *)abcdk_heap_clone(src, strlen(src));
    if (!path)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

    find = dirname(path);
    if (find)
        memcpy(dst, find, strlen(find) + 1);

    abcdk_heap_freep((void**)&path);

    return dst;
}

char *abcdk_basename(char *dst, const char *src)
{
    char *find = NULL;
    char *path = NULL;

    assert(dst != NULL && src != NULL);

    path = (char *)abcdk_heap_clone(src, strlen(src));
    if (!path)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

    find = basename(path);
    if (find)
        memcpy(dst, find, strlen(find) + 1);

    abcdk_heap_freep((void**)&path);

    return dst;
}

char *abcdk_abspath(char *buf, size_t decrease)
{
    abcdk_tree_t *stack = NULL, *pos = NULL;
    const char *p, *p_next;

    /*准备堆栈。*/
    stack = abcdk_tree_alloc3(1);
    if (!stack)
        return NULL;

    /*拆分路径。*/
    p_next = buf;

    while (1)
    {
        p = abcdk_strtok(&p_next, "/");
        if (!p)
            break;

        /*“.”表示当前目录。*/
        if (abcdk_strncmp(p, ".", p_next - p, 1) == 0)
            continue;

        /*“..”表示上一层目录，这里要从堆栈中删除上一层。*/
        if (abcdk_strncmp(p, "..", p_next - p, 1) == 0)
        {
            pos = abcdk_tree_child(stack, 0);
            if (pos)
            {
                abcdk_tree_unlink(pos);
                abcdk_tree_free(&pos);
            }

            continue;
        }

        pos = abcdk_tree_alloc3(p_next - p + 1);
        if (!pos)
            goto final_error;

        strncpy(pos->obj->pstrs[0], p, p_next - p);
        abcdk_tree_insert2(stack, pos, 0);
    }

    /*清空旧路径。*/
    for (int i = 0; buf[i]; i++)
    {
        if (i == 0 && buf[i] == '/')
            continue;

        buf[i] = 0;
    }

    /*缩减深度。*/
    for (int i = 0; i < decrease; i++)
    {
        pos = abcdk_tree_child(stack, 0);
        if (!pos)
            break;

        abcdk_tree_unlink(pos);
        abcdk_tree_free(&pos);
    }

    /*拼装路径。*/
    pos = abcdk_tree_child(stack, 1);
    while (pos)
    {
        abcdk_dirdir(buf, pos->obj->pstrs[0]);
        pos = abcdk_tree_sibling(pos, 0);
    }

    abcdk_tree_free(&stack);
    return buf;

final_error:

    abcdk_tree_free(&stack);
    return NULL;
}
