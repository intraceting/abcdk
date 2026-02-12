/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#include "abcdk/util/iconv.h"

#ifdef _ICONV_H

ssize_t abcdk_iconv(iconv_t cd, const char *src, size_t slen, char *dst, size_t dlen,size_t *remain)
{
    ssize_t chk = 0;
    char *copy_src = NULL;
    char *copy_dst = NULL;
    size_t copy_slen = 0;
    size_t copy_dlen = 0;

    assert(cd != NULL && src != NULL && slen > 0 && dst != NULL && dlen > 0);

    copy_src = (char*)src;
    copy_dst = (char*)dst;
    copy_slen = slen;
    copy_dlen = dlen;

    chk = iconv(cd, &copy_src, &copy_slen, &copy_dst, &copy_dlen);
    if(chk == -1)
        return -1;

    /* 调用者可能关注未转换成功的.*/    
    if(remain != NULL)
        *remain = chk;
    
    return dlen - copy_dlen;
}

ssize_t abcdk_iconv2(const char *from,const char *to, const char *src, size_t slen, char *dst, size_t dlen,size_t *remain)
{
    iconv_t cd = 0;
    size_t ret = -1;

    assert(from != NULL && to != NULL);

    assert(*from != '\0' && *to != '\0');

    cd = iconv_open(to,from);

    if(cd != (iconv_t)-1)
    {
        ret = abcdk_iconv(cd,src,slen,dst,dlen,remain);
        iconv_close(cd);
    }

    return ret;
}

#else //_ICONV_H

ssize_t abcdk_iconv(iconv_t cd, const char *src, size_t slen, char *dst, size_t dlen,size_t *remain)
{
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含ICONV工具."));
    return -1;
}

ssize_t abcdk_iconv2(const char *from,const char *to, const char *src, size_t slen, char *dst, size_t dlen,size_t *remain)
{
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含ICONV工具."));
    return -1;
}

#endif //_ICONV_H