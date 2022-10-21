/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/string.h"

int abcdk_isodigit(int c)
{
    return ((c >= '0' && c <= '7') ? 1 : 0);
}

char *abcdk_strdup(const char *str)
{
    assert(str != NULL);

    return (char*)abcdk_heap_clone(str,strlen(str));
}

const char *abcdk_strstr(const char *str, const char *sub, int caseAb)
{
    assert(str != NULL && sub != NULL);

    if (caseAb)
        return strstr(str, sub);

    return strcasestr(str, sub);
}

const char* abcdk_strstr_eod(const char *str, const char *sub,int caseAb)
{
    const char *addr = NULL;

    assert(str != NULL && sub != NULL);

    addr = abcdk_strstr(str,sub,caseAb);
    
    if(addr)
        addr += strlen(sub);

    return addr;
}

int abcdk_strcmp(const char *s1, const char *s2, int caseAb)
{
    assert(s1 != NULL && s2 != NULL);

    if (caseAb)
        return strcmp(s1, s2);

    return strcasecmp(s1, s2);
}

int abcdk_strncmp(const char *s1, const char *s2, size_t len, int caseAb)
{
    assert(s1 != NULL && s2 != NULL && len > 0);

    if (caseAb)
        return strncmp(s1, s2, len);

    return strncasecmp(s1, s2, len);
}

char *abcdk_strtrim(char *str, int (*isctype_cb)(int c),int where)
{
    char *tmp = NULL;
    size_t len = 0;
    size_t blklen = 0;

    assert(str && isctype_cb);

    tmp = str;
    len = strlen(str);

    if (len <= 0)
        goto final;

    if (0 == where)
    {
        while (*tmp)
            tmp++;

        while (tmp-- > str)
        {
            if(isctype_cb(*tmp))
                *tmp = '\0';
            else 
                goto final;
        }
    }
    else if (1 == where)
    {
        while (*tmp && isctype_cb(*tmp))
        {
            tmp++;
            blklen++;
        }

        if (blklen <= 0)
            goto final;

        for (size_t i = 0; i < len - blklen; i++)
            str[i] = str[i + blklen];

        for (size_t j = len - blklen; j < len; j++)
            str[j] = '\0';
    }
    else if (2 == where)
    {
        abcdk_strtrim(str,isctype_cb,0);
        abcdk_strtrim(str,isctype_cb,1);
    }

final:

    return str;
}

char *abcdk_strtok(char *str,const char *delim, char **saveptr)
{
    char* prev = NULL;
    char* find = NULL;

    assert(str && delim && saveptr);

    if(*saveptr)
        prev = *saveptr;
    else 
        prev = str;

    find = (char *)abcdk_strstr(prev, delim, 1);
    if (find)
    {
        *find = '\0';
        *saveptr = find + strlen(delim);
    }
    else if (*prev != '\0')
    {
        *saveptr = prev + strlen(prev);
    }
    else
    {
        prev = NULL;
        *saveptr = NULL;
    }

    return prev;
}

int abcdk_strtype(const char* str,int (*isctype_cb)(int c))
{
    const char* s = NULL;

    assert(str && isctype_cb);

    s = str;

    if(*s == '\0')
        return 0;

    while(*s)
    {
        if(!isctype_cb(*s++))
            return 0;
    }

    return 1;
}

char *abcdk_strrep(const char *str, const char *src, const char *dst, int caseAb)
{
    size_t srclen = 0, dstlen = 0, str2len = 0,skiplen = 0;
    char *str2 = NULL, *tmp = NULL;
    const char *s = NULL, *e = NULL;

    assert(str != NULL && src != NULL && dst != NULL);

    srclen = strlen(src);
    dstlen = strlen(dst);

    s = str;

    while (s && *s)
    {
        e = abcdk_strstr(s, src, caseAb);
        if (e)
        {
            skiplen = (e - s) + dstlen;
            tmp = abcdk_heap_realloc(str2, str2len + skiplen + 1);
            if (!tmp)
                goto final_error;
            str2 = tmp;

            /*Copy.*/
            strncpy(str2 + str2len, s, e - s);
            strncpy(str2 + str2len + (e - s), dst, dstlen);

            /**/
            str2len += skiplen;

            /*Continue.*/
            s = e + srclen;
        }
        else
        {
            skiplen = strlen(str) - (s - str); //=strlen(s)
            tmp = abcdk_heap_realloc(str2, str2len + skiplen + 1);
            if (!tmp)
                goto final_error;
            str2 = tmp;

            /*Copy.*/
            strcpy(str2 + str2len, s);

            /**/
            str2len += skiplen;

            /*End.*/
            s = NULL;
        }
    }

    return str2;

final_error:

    abcdk_heap_free(str2);

    return NULL;
}

int abcdk_fnmatch(const char *str,const char *wildcard,int caseAb,int ispath)
{
    int flag = 0;
    int chk = FNM_NOMATCH;

    assert(str && wildcard);

    if (!caseAb)
        flag |= FNM_CASEFOLD;
    if (ispath)
        flag |= FNM_PATHNAME | FNM_PERIOD;

    chk = fnmatch(wildcard, str, flag);

    return ((chk==FNM_NOMATCH)?-1:0);
}

size_t abcdk_cslen(const void *str, int width)
{
    size_t len = 0;

    assert(str != NULL);
    assert(width == 1 || width == 2 || width == 4);

    if (width == 4)
    {
        while (ABCDK_PTR2U32(str, len))
            len += 1;
    }
    else if (width == 2)
    {
        while (ABCDK_PTR2U16(str, len))
            len += 1;
    }
    else if (width == 1)
    {
        while (ABCDK_PTR2U8(str, len))
            len += 1;
    }

    return len;
}
