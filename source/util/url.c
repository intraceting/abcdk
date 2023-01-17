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

abcdk_object_t *abcdk_url_encode2(const char *src, size_t slen, int component)
{
    abcdk_object_t *dst = NULL;

    assert(src != NULL && slen > 0);

    dst = abcdk_object_alloc2(slen * 3);
    if (!dst)
        return NULL;

    abcdk_url_encode(src, slen, dst->pstrs[0], &dst->sizes[0], component);

    return dst;
}

ssize_t abcdk_url_decode(const char *src,size_t slen,char *dst,size_t *dlen,int qm_stop)
{
    int s = 0, d = 0;
    int tmp;

    assert(src != NULL && slen > 0 && dst != NULL && dlen != NULL && *dlen > 0);

    for (; s < slen && d < *dlen;)
    {
        if (qm_stop && src[s] == '?')
            break;

        if (src[s] != '%')
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

abcdk_object_t *abcdk_url_decode2(const char *src,size_t slen,int qm_stop)
{
    abcdk_object_t *dst = NULL;

    assert(src != NULL && slen > 0);

    dst = abcdk_object_alloc2(slen);
    if(!dst)
        return NULL;

    abcdk_url_decode(src,slen,dst->pstrs[0],&dst->sizes[0],qm_stop);

    return dst;
}

char *abcdk_url_abspath(char *buf, size_t decrease)
{
    const char *p1 = NULL, *p2 = NULL;

    assert(buf != NULL);

    p1 = abcdk_strstr_eod(buf, "://", 1);
    if (!p1)
        return abcdk_abspath(buf, decrease);

    p2 = abcdk_strstr(p1, "/", 1);
    if (p2)
        return abcdk_abspath((char *)p2, decrease);

    return buf;
}

abcdk_object_t *abcdk_url_fixpath(const char *target, const char *opaque)
{
    abcdk_object_t *dst = NULL;
    size_t dlen = 0;
    const char *p1 = NULL, *p2 = NULL;

    assert(target != NULL && opaque != NULL);

    p1 = abcdk_strstr_eod(target, "://", 1);
    if (p1)
    {
        dst = abcdk_object_alloc_copyfrom(target, strlen(target));
        goto final;
    }

    dlen += strlen(opaque);
    dlen += strlen(target);

    dst = abcdk_object_alloc2(dlen + 2);
    if (!dst)
        return NULL;

    p1 = abcdk_strstr_eod(opaque, "://", 1);
    if (p1)
    {
        if (*target == '/')
            p2 = abcdk_strstr_eod(p1, "/", 1);

        if (p2)
            strncat(dst->pstrs[0], opaque, p2 - opaque);
        else
            strcat(dst->pstrs[0], opaque);

        strcat(dst->pstrs[0], "/");
        strcat(dst->pstrs[0], target);
    }
    else
    {
        if(*target == '/')
        {
            abcdk_dirdir(dst->pstrs[0], target);
        }
        else
        {
            abcdk_dirdir(dst->pstrs[0], opaque);
            abcdk_dirdir(dst->pstrs[0], target);
        }
    }

final:

    abcdk_url_abspath(dst->pstrs[0],0);
    dst->sizes[0] = strlen(dst->pstrs[0]);

    return dst;
}
