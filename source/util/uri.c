/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/uri.h"

abcdk_object_t *abcdk_uri_split(const char *uri)
{
    const char* mark = NULL;
    const char* a_mark = NULL;
    size_t len = 0;
    size_t sizes[5] = {0};
    abcdk_object_t *alloc = NULL;

    assert(uri != NULL);
    assert(uri[0] != '\0');

    len = strlen(uri);

    mark = abcdk_strstr(uri,"://",0);
    if (mark)
    {
        sizes[ABCDK_URI_SCHEME] = 64;
        sizes[ABCDK_URI_USER] = 64;
        sizes[ABCDK_URI_PSWD] = 128;
        sizes[ABCDK_URI_HOST] = 255;
        sizes[ABCDK_URI_PATH] = len + 1;
    }
    else
    {
        sizes[ABCDK_URI_SCHEME] = sizes[ABCDK_URI_USER] = 1;
        sizes[ABCDK_URI_PSWD] = sizes[ABCDK_URI_HOST] = 1;
        sizes[ABCDK_URI_PATH] = len + 1; //set.
    }

    alloc = abcdk_object_alloc(sizes,ABCDK_ARRAY_SIZE(sizes),0);
    if(!alloc)
        goto final;
    
    if(mark)
    {
        if(ABCDK_PTR2I8(mark,3) == '/')
        {
            /* SCHEME:///abcdk/...*/
            sscanf(uri,"%[^:]%*3[:/]%s",alloc->pptrs[ABCDK_URI_SCHEME],alloc->pptrs[ABCDK_URI_PATH]);
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
                sscanf(uri, "%[^:]%*3[:/]%[^:]%*1[:]%[^@]%*1[@]%[^/]%s",
                       alloc->pptrs[ABCDK_URI_SCHEME], alloc->pptrs[ABCDK_URI_USER],
                       alloc->pptrs[ABCDK_URI_PSWD], alloc->pptrs[ABCDK_URI_HOST],
                       alloc->pptrs[ABCDK_URI_PATH]);
            }
            else
            {
                /* 
                 * SCHEME://host[:port]/abcdk
                 * SCHEME://[host][:port]/abcdk
                */
                sscanf(uri, "%[^:]%*3[:/]%[^/]%s",
                       alloc->pptrs[ABCDK_URI_SCHEME], alloc->pptrs[ABCDK_URI_HOST],
                       alloc->pptrs[ABCDK_URI_PATH]);
            }
        }
    }
    else
    {
        memcpy(alloc->pptrs[ABCDK_URI_PATH],uri,len);
    }

final:

    return alloc;
}

int _abcdk_uri_encode_check_escape(int c, int component)
{
    if (c >= 'A' && c <= 'Z')
        return 1;
    if (c >= 'a' && c <= 'z')
        return 1;
    if (c >= '0' && c <= '9')
        return 1;

    if (component)
    {
        static char dict[] = {"!’()*-._~"};
        for (int i = 0; i < sizeof(dict); i++)
        {
            if (c == dict[i])
                return 1;
        }
    }
    else
    {
        static char dict[] = {"!#$&’()*+,/:;=?@-._~"};
        for (int i = 0; i < sizeof(dict); i++)
        {
            if (c == dict[i])
                return 1;
        }
    }

    return 0;
}

ssize_t abcdk_uri_encode(const char *src, size_t slen, char *dst, size_t *dlen, int component)
{
    int s = 0, d = 0;

    assert(src != NULL && slen > 0 && dst != NULL && dlen != NULL && *dlen > 0);

    for (; s < slen && d < *dlen; s++)
    {
        if (_abcdk_uri_encode_check_escape(src[s],component))
        {
            dst[d] = src[s];
            d += 1;
        }
        else
        {
            /*不通超过密文可用空间。*/
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

ssize_t abcdk_uri_decode(const char *src,size_t slen,char *dst,size_t *dlen)
{
    int s = 0, d = 0;
    int tmp;

    assert(src != NULL && slen > 0 && dst != NULL && dlen != NULL && *dlen > 0);

    for (; s < slen && d < *dlen;)
    {
        if (src[s] == '+')
        {
            dst[d] = ' ';
            d += 1;
            s += 1;
        }
        else if (src[s] != '%')
        {
            dst[d] = src[s];
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