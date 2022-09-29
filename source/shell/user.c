/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/shell/user.h"

char *abcdk_user_dirname(char *buf, const char *append)
{
    assert(buf);

    snprintf(buf, PATH_MAX, "/var/run/user/%d/", getuid());

    if (access(buf, R_OK | W_OK | X_OK | F_OK) != 0)
        ABCDK_ERRNO_AND_RETURN1(ENOENT, NULL);

    if (append)
        abcdk_dirdir(buf, append);

    return buf;
}
