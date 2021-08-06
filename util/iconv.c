/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk-util/iconv.h"

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

    /* 调用者可能关注未转换成功的。*/    
    if(remain != NULL)
        *remain = chk;
    
    return dlen - copy_dlen;
}
