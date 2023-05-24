/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/url.h"

void _abcdk_url_split_free_cb(abcdk_object_t *obj, void *opaque)
{
    if(!obj)
        return;

    for (int i = 0; i < obj->numbers; i++)
        abcdk_heap_free(obj->pstrs[i]);
}

abcdk_object_t *abcdk_url_split(const char *url)
{
    abcdk_object_t *obj = NULL;
    const char *p = NULL ,*p_next = NULL;
    const char *p2 = NULL ,*p2_next = NULL;
    
    assert(url != NULL);

    obj = abcdk_object_alloc(NULL,10,0);
    if(!obj)
        return NULL;

    abcdk_object_atfree(obj,_abcdk_url_split_free_cb,NULL);

    p_next = url;

    p = abcdk_strstr(p_next, "://", 0);
    if (!p)
        goto path_split;

    if (p != p_next)
    {
        p = abcdk_strtok(&p_next, "://");
        if (!p)
            goto final;

        obj->pstrs[ABCDK_URL_SCHEME] = abcdk_heap_clone(p, p_next - p);
        obj->sizes[ABCDK_URL_SCHEME] = strlen(obj->pstrs[ABCDK_URL_SCHEME]);
    }

    if (*p_next == ':')
    {
        obj->pstrs[ABCDK_URL_FLAG] = abcdk_heap_clone("://", 3);
        obj->sizes[ABCDK_URL_FLAG] = 3;
        p_next += 3;
    }

    /*查找HOST:PORT。*/
    p = abcdk_strtok(&p_next, "/");
    if (!p)
        goto final;

    p2 = abcdk_strstr(p, "@", 0);
    if (p2)
    {
        p_next = p;
        p = abcdk_strtok(&p_next, "@");
        if (!p)
            goto final;

        obj->pstrs[ABCDK_URL_AUTH] = abcdk_heap_clone(p, p_next - p);
        obj->sizes[ABCDK_URL_AUTH] = strlen(obj->pstrs[ABCDK_URL_AUTH]);
        
        if (*p_next == '@')
            p_next += 1;

        /*查找HOST:PORT。*/
        p = abcdk_strtok(&p_next, "/");
        if (!p)
            goto final;

        /*拆分用户名和密码。*/
        
        p2_next = obj->pstrs[ABCDK_URL_AUTH];
        p2 = abcdk_strtok(&p2_next, ":");
        if (!p2)
            goto host_split;

        obj->pstrs[ABCDK_URL_USER] = abcdk_heap_clone(p2, p2_next - p2);
        obj->sizes[ABCDK_URL_USER] = strlen(obj->pstrs[ABCDK_URL_USER]);

        if (*p2_next == ':')
            p2_next += 1;

        p2 = abcdk_strtok(&p2_next, "@");
        if (!p2)
            goto host_split;

        obj->pstrs[ABCDK_URL_PSWD] = abcdk_heap_clone(p2, p2_next - p2);
        obj->sizes[ABCDK_URL_PSWD] = strlen(obj->pstrs[ABCDK_URL_PSWD]);
    }

host_split:

    if (*p_next == '/')
    {
        obj->pstrs[ABCDK_URL_HOST] = abcdk_heap_clone(p, p_next - p);
        obj->sizes[ABCDK_URL_HOST] = strlen(obj->pstrs[ABCDK_URL_HOST]);
    }
    else
    {
        /*可能域名后面直接跟锚点。*/
        p_next = p;
        p = abcdk_strtok(&p_next, "#");
        if (!p)
            goto final;

        obj->pstrs[ABCDK_URL_HOST] = abcdk_heap_clone(p, p_next - p);
        obj->sizes[ABCDK_URL_HOST] = strlen(obj->pstrs[ABCDK_URL_HOST]);
    }

path_split:

    p = abcdk_strtok(&p_next, "\r\n");
    if (!p)
        goto final;

    obj->pstrs[ABCDK_URL_SCRIPT] = abcdk_heap_clone(p, p_next - p);
    obj->sizes[ABCDK_URL_SCRIPT] = strlen(obj->pstrs[ABCDK_URL_SCRIPT]);

    p_next = obj->pstrs[ABCDK_URL_SCRIPT];

    /*可能没有参数，这里要判断一下。*/
    p2 = abcdk_strstr(p_next, "?", 0);
    if (p2)
    {
        p = p_next;
        p_next = p2;

        obj->pstrs[ABCDK_URL_PATH] = abcdk_heap_clone(p, p_next - p);
        obj->sizes[ABCDK_URL_PATH] = strlen(obj->pstrs[ABCDK_URL_PATH]);

        if (*p_next == '?')
            p_next += 1;

        p = abcdk_strtok(&p_next, "#");
        if (!p)
            goto final;

        obj->pstrs[ABCDK_URL_PARAM] = abcdk_heap_clone(p, p_next - p);
        obj->sizes[ABCDK_URL_PARAM] = strlen(obj->pstrs[ABCDK_URL_PARAM]);
    }
    else
    {
        p = abcdk_strtok(&p_next, "#");
        if (!p)
            goto final;

        obj->pstrs[ABCDK_URL_PATH] = abcdk_heap_clone(p, p_next - p);
        obj->sizes[ABCDK_URL_PATH] = strlen(obj->pstrs[ABCDK_URL_PATH]);
    }

    if (*p_next == '#')
        p_next += 1;

    p = abcdk_strtok(&p_next, "\r\n");
    if (!p)
        goto final;

    obj->pstrs[ABCDK_URL_ANCHOR] = abcdk_heap_clone(p, p_next - p);
    obj->sizes[ABCDK_URL_ANCHOR] = strlen(obj->pstrs[ABCDK_URL_ANCHOR]);

final:

    return obj;
}

abcdk_object_t *abcdk_url_create(int max, const char *fmt, ...)
{
    abcdk_object_t *buf = NULL,*url = NULL;

    assert(max > 0 && fmt != NULL);

    va_list ap;
    va_start(ap, fmt);
    buf = abcdk_object_vprintf(max,fmt,ap);
    va_end(ap);

    if(!buf)
        return NULL;

    url = abcdk_url_split(buf->pstrs[0]);
    abcdk_object_unref(&buf);

    return url;
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

abcdk_object_t *abcdk_url_decode2(const char *src, size_t slen, int qm_stop)
{
    abcdk_object_t *dst = NULL;

    assert(src != NULL && slen > 0);

    dst = abcdk_object_alloc2(slen + 1);
    if (!dst)
        return NULL;

    abcdk_url_decode(src, slen, dst->pstrs[0], &dst->sizes[0], qm_stop);

    return dst;
}

char *abcdk_url_abspath(char *buf, size_t decrease)
{
    const char *p1 = NULL, *p2 = NULL;

    assert(buf != NULL);

    p1 = abcdk_strstr_eod(buf, "://", 1);
    if (p1)
    {
        p2 = abcdk_strstr(p1, "/", 1);
        if (p2)
            abcdk_abspath((char *)p2, decrease);
    }
    else
    {
        abcdk_abspath(buf, decrease);
    }

    return buf;
}

abcdk_object_t *abcdk_url_fixpath(const char *target, const char *opaque)
{
    abcdk_object_t *u1 = NULL, *u2 = NULL;
    abcdk_object_t *dst = NULL;
    size_t dlen = 0;
    const char *p1 = NULL, *p2 = NULL;

    assert(target != NULL && opaque != NULL);

    dlen += strlen(opaque);
    dlen += strlen(target);

    dst = abcdk_object_alloc2(dlen + 2);
    if (!dst)
        return NULL;

    u1 = abcdk_url_split(target);
    u2 = abcdk_url_split(opaque);
    if (!u1 || !u2)
        goto final;

    if (u1->pstrs[ABCDK_URL_SCHEME])
        strcat(dst->pstrs[0], u1->pstrs[ABCDK_URL_SCHEME]);
    else if (u2->pstrs[ABCDK_URL_SCHEME])
        strcat(dst->pstrs[0], u2->pstrs[ABCDK_URL_SCHEME]);

    if (u1->pstrs[ABCDK_URL_FLAG])
        strcat(dst->pstrs[0], u1->pstrs[ABCDK_URL_FLAG]);
    else if (u2->pstrs[ABCDK_URL_FLAG])
        strcat(dst->pstrs[0], u2->pstrs[ABCDK_URL_FLAG]);

    if (u1->pstrs[ABCDK_URL_AUTH])
    {
        strcat(dst->pstrs[0], u1->pstrs[ABCDK_URL_AUTH]);
        strcat(dst->pstrs[0], "@");
    }
    else if (u2->pstrs[ABCDK_URL_AUTH])
    {
        strcat(dst->pstrs[0], u2->pstrs[ABCDK_URL_AUTH]);
        strcat(dst->pstrs[0], "@");
    }

    if (u1->pstrs[ABCDK_URL_HOST])
        strcat(dst->pstrs[0], u1->pstrs[ABCDK_URL_HOST]);
    else if (u2->pstrs[ABCDK_URL_HOST])
        strcat(dst->pstrs[0], u2->pstrs[ABCDK_URL_HOST]);

    if (u1->pstrs[ABCDK_URL_PATH])
    {
        if (u1->pstrs[ABCDK_URL_PATH][0] != '/' && u2->pstrs[ABCDK_URL_PATH])
            strcat(dst->pstrs[0], u2->pstrs[ABCDK_URL_PATH]);

        strcat(dst->pstrs[0], "/");
        strcat(dst->pstrs[0], u1->pstrs[ABCDK_URL_PATH]);
    }

    if (u1->pstrs[ABCDK_URL_PARAM])
    {
        strcat(dst->pstrs[0], "?");
        strcat(dst->pstrs[0], u1->pstrs[ABCDK_URL_PARAM]);
    }

    if (u1->pstrs[ABCDK_URL_ANCHOR])
    {
        strcat(dst->pstrs[0], "#");
        strcat(dst->pstrs[0], u1->pstrs[ABCDK_URL_ANCHOR]);
    }

final:

    abcdk_url_abspath(dst->pstrs[0],0);
    dst->sizes[0] = strlen(dst->pstrs[0]);

    abcdk_object_unref(&u1);
    abcdk_object_unref(&u2);

    return dst;
}
