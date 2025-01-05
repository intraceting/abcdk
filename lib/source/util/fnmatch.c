/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/util/fnmatch.h"

int abcdk_fnmatch(const char *str,const char *pattern,int caseAb,int ispath)
{
    abcdk_object_t *p;
    int flag = 0;
    int chk = FNM_NOMATCH;

    assert(str && pattern);

    if (!caseAb)
        flag |= FNM_CASEFOLD;
    if (ispath)
        flag |= FNM_PATHNAME;

    if(flag & FNM_PATHNAME)
    {
        p = abcdk_object_alloc3(PATH_MAX+1,2);
        if(!p)
            return -2;
        
        strncpy(p->pstrs[0],str,p->sizes[0]);
        strncpy(p->pstrs[1],pattern,p->sizes[1]);
        abcdk_abspath(p->pstrs[0],0);
        abcdk_abspath(p->pstrs[1],0);

        chk = fnmatch(p->pstrs[1], p->pstrs[0], flag);

        abcdk_object_unref(&p);
    }
    else 
    {
        chk = fnmatch(pattern, str, flag);
    }

    return ((chk==FNM_NOMATCH)?-1:0);
}