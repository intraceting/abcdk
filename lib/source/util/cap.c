/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#include "abcdk/util/cap.h"

#ifdef _SYS_CAPABILITY_H

int abcdk_cap_get_pid(pid_t pid, cap_value_t power, cap_flag_t flag)
{
    cap_t cap_p = NULL;
    cap_flag_value_t v = 0;
    int chk;

    assert(pid >= 0);

    cap_p = cap_get_pid(pid);
    if (!cap_p)
        return -1;

    chk = cap_get_flag(cap_p, power, flag, &v);
    if (chk != 0)
    {
        chk = -1;
        goto final;
    }

    /*Check power.*/
    chk = ((v == CAP_SET) ? 1 : ((v == CAP_CLEAR) ? 0 : -1));

final:

    cap_free(cap_p);

    return chk;
}

int abcdk_cap_set_pid(pid_t pid,cap_value_t power, cap_flag_t flag,cap_flag_value_t cmd)
{
    cap_t cap_p = NULL;
    cap_flag_value_t v = 0;
    int chk;

    assert(pid >= 0);

    cap_p = cap_get_pid(pid);
    if (!cap_p)
        return -1;

    chk = cap_set_flag(cap_p,flag,1,&power, cmd);
    if (chk != 0)
    {
        chk = -1;
        goto final;
    }

    chk = cap_set_proc(cap_p);
    if (chk != 0)
    {
        chk = -1;
        goto final;
    }
    
    /*No error.*/
    chk = 0;

final:

    cap_free(cap_p);

    return chk;
}

#endif //_SYS_CAPABILITY_H