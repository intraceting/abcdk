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

int _abcdk_strtrim_check(int c, int (*isctype_cb)(int c), const char *other)
{
    if(isctype_cb)
    {
        if(isctype_cb(c))
            return 1;
    }

    if (other)
    {
        for (; *other; other++)
        {
            if (c == *other)
                return 1;
        }
    }

    return 0;
}

char* abcdk_strtrim(char* str,int (*isctype_cb)(int c),int where)
{
    return abcdk_strtrim2(str,isctype_cb,NULL,where);
}

char *abcdk_strtrim2(char *str, int (*isctype_cb)(int c), const char *other,int where)
{
    char *tmp = NULL;

    assert(str && (isctype_cb || other) );

    tmp = str;

    if (!*tmp)
        goto final;

    if (0 == where)
    {
        while (*tmp)
            tmp++;

        while (tmp-- > str)
        {
            if(_abcdk_strtrim_check(*tmp,isctype_cb,other))
                *tmp = '\0';
            else 
                goto final;
        }
    }
    else if (1 == where)
    {
        while (*tmp)
        {
            if (!_abcdk_strtrim_check(*tmp, isctype_cb, other))
                break;
            tmp++;
        }

        for (int i = 0; ; i++)
        {
            str[i] = *tmp++;
            if(!str[i])
                goto final;
        }
    }
    else if (2 == where)
    {
        abcdk_strtrim2(str,isctype_cb,other,0);
        abcdk_strtrim2(str,isctype_cb,other,1);
    }

final:

    return str;
}

const char *abcdk_strtok(const char **next, const char *delim)
{
    return abcdk_strtok2(next,delim,0);
}

const char *abcdk_strtok2(const char **next, const char *delim, int skip_space)
{
    const char *start_p = NULL,*find_p = NULL;
    int dlen = 0;

    assert(next != NULL && delim != NULL);
    assert(*next != NULL && *delim != '\0');

    start_p = *next;
    dlen = strlen(delim);

NEXT_SEGMENT:

    if(*start_p == '\0')
        return NULL;

    if (skip_space)
    {
        for (; *start_p; start_p++)
        {
            if (!isspace(*start_p))
                break;
        }
    }
  
    find_p = abcdk_strstr(start_p, delim, 1);
    if (!find_p)
    {
        /*未找到分割符，定位到字符串末尾。*/
        find_p = start_p;
        while (*find_p)
            find_p++;
    }
    else if (find_p == start_p)
    {
        start_p += dlen;
        goto NEXT_SEGMENT;
    }

    *next = find_p;
    return start_p; 
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
