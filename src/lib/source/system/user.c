/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/system/user.h"

char *abcdk_user_dir_run(char *buf, const char *append)
{
    assert(buf);

    snprintf(buf, PATH_MAX, "/var/run/user/%d/", getuid());

    if (append)
        abcdk_dirdir(buf, append);

    return buf;
}

char *abcdk_user_dir_home(char *buf, const char *append)
{
    char *home_p = NULL;

    assert(buf);

    home_p = getenv("HOME");
    if (home_p)
        abcdk_dirdir(buf, home_p);

    if (append)
        abcdk_dirdir(buf, append);

    return buf;
}
