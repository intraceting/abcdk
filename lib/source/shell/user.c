/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/shell/user.h"

char *abcdk_user_dirname(char *buf, const char *append)
{
    assert(buf);

    snprintf(buf, PATH_MAX, "/var/run/user/%d/", getuid());

    if (access(buf, F_OK) != 0)
        return NULL;

    if (append)
        abcdk_dirdir(buf, append);

    return buf;
}
