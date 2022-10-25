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
        if ((path[len - 1] == '/') && (suffix[0] == '/'))
        {
            path[len - 1] = '\0';
            len -= 1;
        }
        else if ((path[len - 1] != '/') && (suffix[0] != '/'))
        {
            path[len] = '/';
            len += 1;
        }
    }

    /* 要有足够的可用空间，不然会溢出。 */
    strcat(path + len, suffix);

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
        abcdk_heap_free2((void**)&tmp);
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

    abcdk_heap_free2((void**)&path);

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

    abcdk_heap_free2((void**)&path);

    return dst;
}

char *abcdk_abspath(char *buf, const char *file, const char *path)
{
    assert(buf != NULL && file != NULL);

    if (file[0] != '/')
    {
        if (path && path[0])
            abcdk_dirdir(buf,path);
        else
            getcwd(buf, PATH_MAX);
    }

    abcdk_dirdir(buf,file);

    return buf;
}
