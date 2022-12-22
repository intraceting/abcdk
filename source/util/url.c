/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/url.h"

abcdk_object_t *abcdk_url_split(const char *url)
{
    const char* mark = NULL;
    const char* a_mark = NULL;
    size_t len = 0;
    size_t sizes[5] = {0};
    abcdk_object_t *obj = NULL;

    assert(url != NULL);
    assert(url[0] != '\0');

    len = strlen(url);

    mark = abcdk_strstr(url,"://",0);
    if (mark)
    {
        sizes[ABCDK_URL_SCHEME] = 64;
        sizes[ABCDK_URL_USER] = 64;
        sizes[ABCDK_URL_PSWD] = 128;
        sizes[ABCDK_URL_HOST] = 255;
        sizes[ABCDK_URL_PATH] = len + 1;
    }
    else
    {
        sizes[ABCDK_URL_SCHEME] = sizes[ABCDK_URL_USER] = 1;
        sizes[ABCDK_URL_PSWD] = sizes[ABCDK_URL_HOST] = 1;
        sizes[ABCDK_URL_PATH] = len + 1; //set.
    }
    
    obj = abcdk_object_alloc(sizes,ABCDK_ARRAY_SIZE(sizes),0);
    if(!obj)
        return NULL;
    
    if(mark)
    {
        if(ABCDK_PTR2I8(mark,3) == '/')
        {
            /* SCHEME:///abcdk/...*/
            sscanf(url,"%[^:]%*3[:/]%s",obj->pstrs[ABCDK_URL_SCHEME],obj->pstrs[ABCDK_URL_PATH]);
        }
        else
        {
            for (size_t k = 3; ABCDK_PTR2I8(mark, k); k++)
            {
                if (ABCDK_PTR2I8(mark, k) == '/')
                    break;

                if (ABCDK_PTR2I8(mark, k) == '@')
                    a_mark = ABCDK_PTR2I8PTR(mark, k);
            }

            if(a_mark)
            {
                /* 
                 * SCHEME://user:pswd@host[:port]/abcdk
                 * SCHEME://user:pswd@[host][:port]/abcdk
                */
                sscanf(url, "%[^:]%*3[:/]%[^:]%*1[:]%[^@]%*1[@]%[^/]%s",
                       obj->pstrs[ABCDK_URL_SCHEME], obj->pstrs[ABCDK_URL_USER],
                       obj->pstrs[ABCDK_URL_PSWD], obj->pstrs[ABCDK_URL_HOST],
                       obj->pstrs[ABCDK_URL_PATH]);
            }
            else
            {
                /* 
                 * SCHEME://host[:port]/abcdk
                 * SCHEME://[host][:port]/abcdk
                */
                sscanf(url, "%[^:]%*3[:/]%[^/]%s",
                       obj->pstrs[ABCDK_URL_SCHEME], obj->pstrs[ABCDK_URL_HOST],
                       obj->pstrs[ABCDK_URL_PATH]);
            }
        }
    }
    else
    {
        strncpy(obj->pstrs[ABCDK_URL_PATH],url,len);
    }

    return obj;
}

int _abcdk_url_encode_check_escape(uint8_t c, int component)
{
    if (c >= 'A' && c <= 'Z')
        return 1;
    if (c >= 'a' && c <= 'z')
        return 1;
    if (c >= '0' && c <= '9')
        return 1;

    if (component)
    {
        static char dict[] = {"!'()*-._~"};
        for (int i = 0; i < sizeof(dict); i++)
        {
            if (c == (uint8_t)dict[i])
                return 1;
        }
    }
    else
    {
        static char dict[] = {"!#$&'()*+,/:;=?@-._~"};
        for (int i = 0; i < sizeof(dict); i++)
        {
            if (c == (uint8_t)dict[i])
                return 1;
        }
    }

    return 0;
}

ssize_t abcdk_url_encode(const char *src, size_t slen, char *dst, size_t *dlen, int component)
{
    int s = 0, d = 0;

    assert(src != NULL && slen > 0 && dst != NULL && dlen != NULL && *dlen > 0);

    for (; s < slen && d < *dlen; s++)
    {
        if (_abcdk_url_encode_check_escape(src[s],component))
        {
            dst[d] = src[s];
            d += 1;
        }
        else
        {
            /*不能超过密文可用空间。*/
            if (d + 3 > *dlen)
                break;

            sprintf(dst + d, "%%%02X", (uint8_t)src[s]);
            d += 3;
        }
    }

    /*更新密文长度。*/
    *dlen = d;

    return (slen - s);
}

ssize_t abcdk_url_decode(const char *src,size_t slen,char *dst,size_t *dlen,int component)
{
    int s = 0, d = 0, qm = 0;
    int tmp;

    assert(src != NULL && slen > 0 && dst != NULL && dlen != NULL && *dlen > 0);

    for (; s < slen && d < *dlen;)
    {
        /*如果密文是URL，则检测问号(?)位置，并且问号之后的仅做复制。*/
        if (!component && src[s] == '?')
            qm = 1;

        /*如果是URL，则仅转换问号之前的。*/
        if (src[s] != '%' || qm)
        {
            dst[d] = src[s];
            d += 1;
            s += 1;
        }
        else if (src[s] == '+')
        {
            dst[d] = ' ';
            d += 1;
            s += 1;
        }
        else
        {
            tmp = 0;
            sscanf(src + s + 1, "%02x",&tmp);
            dst[d] = tmp;
            d += 1;
            s += 3;
        }
    }

    /*更新明文长度。*/
    *dlen = d;

    return (slen - s);
}