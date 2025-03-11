/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/sdp.h"

void _abcdk_sdp_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    for (int i = 1; i < alloc->numbers; i++)
        abcdk_heap_free(alloc->pstrs[i]);
}

char *_abcdk_sdp_fgetline(FILE *fp, uint8_t delim)
{
    char *line = NULL;
    size_t len = 0;
    ssize_t rlen = 0;
    char *p = NULL;

next_line:

    rlen = abcdk_fgetline(fp, &line, &len, delim, 0);
    if (rlen > 0)
    {
        /*替换换行符。*/
        if (line[rlen - 1] == delim)
            line[rlen - 1] = '\0';

        /*去掉字符串两端所有空白字符。 */
        abcdk_strtrim(line, isspace, 2);

        /*跳过空行。*/
        if (*line == '\0')
            goto next_line;

        p = abcdk_heap_clone(line, strlen(line));
        if (!p)
            goto final;
    }

final:

    if (line)
        free(line);

    return p;
}

void _abcdk_sdp_split(abcdk_tree_t *sdp)
{
    FILE *fp = NULL;
    char *p = NULL;

    p = sdp->obj->pstrs[0];

    fp = fmemopen(p, strlen(p), "r");
    if (!fp)
        goto fianl;

    for (int i = 1; i < 100; i++)
    {
        if (i == 1)
        {
            sdp->obj->pstrs[i] = _abcdk_sdp_fgetline(fp, '=');
            if (!sdp->obj->pstrs[i])
                break;
        }
        else
        {
            if (*p == 'v' || *p == 's' || *p == 'i' || *p == 'u' || *p == 'e' || *p == 'z' || *p == 'r')
            {
                if (i != 2)
                    break;

                sdp->obj->pstrs[i] = _abcdk_sdp_fgetline(fp, '\n');
                if (!sdp->obj->pstrs[i])
                    break;
            }
            else
            {
                sdp->obj->pstrs[i] = _abcdk_sdp_fgetline(fp, ' ');
                if (!sdp->obj->pstrs[i])
                    break;
            }
        }
    }

fianl:

    if (fp)
        fclose(fp);
}

abcdk_tree_t *abcdk_sdp_parse(const char *data, size_t size)
{
    abcdk_tree_t *sdp = NULL, *sub = NULL, *sdp_p = NULL;
    FILE *fp = NULL;

    assert(data != NULL && size > 0);

    fp = fmemopen((void *)data, size, "r");
    if (!fp)
        goto fianl_error;

    sdp = abcdk_tree_alloc3(1);
    if (!sdp)
        goto fianl_error;

    sdp_p = sdp;

    while (1)
    {
        sub = abcdk_tree_alloc2(NULL, 100, 0);
        if (!sub)
            goto fianl_error;

        /*注册清理函数。*/
        abcdk_object_atfree(sub->obj, _abcdk_sdp_destroy_cb, NULL);

        sub->obj->pstrs[0] = _abcdk_sdp_fgetline(fp, '\n');
        if (!sub->obj->pstrs[0])
            break;

        /*分解字段。*/
        _abcdk_sdp_split(sub);

        if (sub->obj->pstrs[0][0] != 'm')
        {
            abcdk_tree_insert2(sdp_p, sub, 0);
        }
        else
        {
            /*指向新的媒体节点。*/
            abcdk_tree_insert2(sdp, sub, 0);
            sdp_p = sub;
        }
    }

    /*OK.*/
    goto fianl;

fianl_error:

    abcdk_tree_free(&sdp);

fianl:

    if (fp)
        fclose(fp);

    return sdp;
}

int _abcdk_sdp_dump_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    FILE *fp = (FILE *)opaque;

    if (depth == 0)
    {
        abcdk_tree_fprintf(fp, depth, node, "SDP\n");
    }
    else if (depth == SIZE_MAX)
    {
        return -1;
    }
    else
    {
        abcdk_tree_fprintf(fp, depth, node, "");
        for (int i = 1; i < 100; i++)
        {
            if (!node->obj->pstrs[i])
                break;

            fprintf(fp, "|");
            fprintf(fp, "%s", node->obj->pstrs[i]);
        }
        fprintf(fp, "|\n");
    }

    return 1;
}

void abcdk_sdp_dump(FILE *fp, abcdk_tree_t *sdp)
{
    abcdk_tree_iterator_t it = {0, fp, _abcdk_sdp_dump_cb};

    abcdk_tree_scan(sdp, &it);
}

abcdk_tree_t *abcdk_sdp_find_media(abcdk_tree_t *sdp, uint8_t fmt)
{
    abcdk_tree_t *p = NULL, *p2 = NULL;

    assert(sdp != NULL && fmt != 0);

    p = abcdk_tree_child(sdp, 1);

    while (p)
    {
        if (p->obj->pstrs[1][0] == 'm')
        {
            /*遍历媒体格式列表，判断当前节点是否包含需要媒体信息。*/
            for (int i = 5; i < 100; i++)
            {
                if (!p->obj->pstrs[i])
                    break;

                if (atoi(p->obj->pstrs[i]) == fmt)
                    return p;
            }
        }

        p = abcdk_tree_sibling(p, 0);
    }

    return NULL;
}

void abcdk_sdp_media_base_free(abcdk_sdp_media_base_t **ctx)
{
    abcdk_sdp_media_base_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->encoder);
    abcdk_object_unref(&ctx_p->encoder_param);
    for (int i = 0; i < 100; i++)
        abcdk_object_unref(&ctx_p->fmtp_param[i]);
    abcdk_object_unref(&ctx_p->control);
    abcdk_object_unref(&ctx_p->sprop_vps);
    abcdk_object_unref(&ctx_p->sprop_sps);
    abcdk_object_unref(&ctx_p->sprop_pps);
    abcdk_object_unref(&ctx_p->sprop_sei);
    abcdk_heap_free(ctx_p);
}

int _abcdk_sdp_media_sprop_decode(abcdk_sdp_media_base_t *ctx, const char *src)
{
    const char *p_next = NULL;
    const char *p = src;

    if (abcdk_strncmp(p, "sprop-parameter-sets=", 21, 0) == 0)
    {
        if (ctx->sprop_sps || ctx->sprop_pps)
            return -1;

        p_next = p + 21;
        p = abcdk_strtok(&p_next, ",");
        if (!p)
            return -1;

        ctx->sprop_sps = abcdk_basecode_decode2(p, p_next - p, 64);
        if (!ctx->sprop_sps)
            return -1;

        p = abcdk_strtok(&p_next, ",");
        if (!p)
            return -1;

        ctx->sprop_pps = abcdk_basecode_decode2(p, p_next - p, 64);
        if (!ctx->sprop_pps)
            return -1;
    }
    else if (abcdk_strncmp(p, "sprop-vps=", 10, 0) == 0)
    {
        if (ctx->sprop_vps)
            return -1;

        p_next = p + 10;
        p = abcdk_strtok(&p_next, ";");
        if (!p)
            return -1;

        ctx->sprop_vps = abcdk_basecode_decode2(p, p_next - p, 64);
        if (!ctx->sprop_vps)
            return -1;
    }
    else if (abcdk_strncmp(p, "sprop-sps=", 10, 0) == 0)
    {
        if (ctx->sprop_sps)
            return -1;

        p_next = p + 10;
        p = abcdk_strtok(&p_next, ";");
        if (!p)
            return -1;

        ctx->sprop_sps = abcdk_basecode_decode2(p, p_next - p, 64);
        if (!ctx->sprop_sps)
            return -1;
    }
    else if (abcdk_strncmp(p, "sprop-pps=", 10, 0) == 0)
    {
        if (ctx->sprop_pps)
            return -1;

        p_next = p + 10;
        p = abcdk_strtok(&p_next, ";");
        if (!p)
            return -1;

        ctx->sprop_pps = abcdk_basecode_decode2(p, p_next - p, 64);
        if (!ctx->sprop_pps)
            return -1;
    }
    else if (abcdk_strncmp(p, "sprop-sei=", 10, 0) == 0)
    {
        if (ctx->sprop_sei)
            return -1;

        p_next = p + 10;
        p = abcdk_strtok(&p_next, ";");
        if (!p)
            return -1;

        ctx->sprop_sei = abcdk_basecode_decode2(p, p_next - p, 64);
        if (!ctx->sprop_sei)
            return -1;
    }

    return 0;
}

abcdk_sdp_media_base_t *abcdk_sdp_media_base_collect(abcdk_tree_t *sdp, uint8_t fmt)
{
    abcdk_sdp_media_base_t *ctx = NULL;
    abcdk_tree_t *a_p = NULL;
    const char *p = NULL, *p_next = NULL;
    uint8_t payload, payload2;
    int chk;

    assert(sdp != NULL);

    /*也许传入的是根节点。*/ 
    if(!abcdk_tree_father(sdp))
    {
        sdp = abcdk_sdp_find_media(sdp,fmt);
        if(!sdp)
            return NULL;
        
        payload = fmt;
    }
    else if (sdp->obj->pstrs[1][0] == 'm')
    {
        /*遍历媒体格式列表，判断当前节点是否包含需要媒体信息。*/
        for (int i = 5; i < 100; i++)
        {
            if (!sdp->obj->pstrs[i])
                return NULL;

            payload = atoi(sdp->obj->pstrs[i]);
            if (payload == fmt)
                break;
        }
    }
    
    ctx = abcdk_heap_alloc(sizeof(abcdk_sdp_media_base_t));
    if (!ctx)
        return NULL;

    /*遍历属性节点。*/
    a_p = abcdk_tree_child(sdp, 1);

    while (a_p)
    {
        if (abcdk_strncmp(a_p->obj->pstrs[2], "rtpmap:", 7, 0) == 0)
        {
            sscanf(a_p->obj->pstrs[2], "%*[^:]%*[:]%hhu", &payload2);
            if (payload2 != payload)
            {
                a_p = abcdk_tree_sibling(a_p, 0);
                continue;
            }

            p_next = a_p->obj->pstrs[3];

            /*拆分编码名称。*/
            p = abcdk_strtok(&p_next, "/");
            if (!p)
                goto final_error;

            ctx->encoder = abcdk_object_copyfrom(p,p_next - p);
            if (!ctx->encoder)
                goto final_error;

            /*拆分时间速率。*/
            p = abcdk_strtok(&p_next, "/");
            if (!p)
                goto final_error;

            sscanf(p, "%u", &ctx->clock_rate);

            /*拆分编码参数(可能不存在)。*/
            p = abcdk_strtok(&p_next, "/");
            if (!p)
            {
                a_p = abcdk_tree_sibling(a_p, 0);
                continue;
            }

            ctx->encoder_param = abcdk_object_copyfrom(p,p_next - p);
            if (!ctx->encoder_param)
                goto final_error;
        }
        else if (abcdk_strncmp(a_p->obj->pstrs[2], "fmtp:", 5, 0) == 0)
        {
            sscanf(a_p->obj->pstrs[2], "%*[^:]%*[:]%hhu", &payload2);
            if (payload2 != payload)
            {
                a_p = abcdk_tree_sibling(a_p, 0);
                continue;
            }

            for (int i = 3,j = 0; i < 100; i++)
            {
                if (!a_p->obj->pstrs[i])
                    break;

                p_next = a_p->obj->pstrs[i];
                while (1)
                {
                    p = abcdk_strtok(&p_next, ";");
                    if(!p)
                        break;

                    ctx->fmtp_param[j] = abcdk_object_copyfrom(p,p_next - p);
                    if (!ctx->fmtp_param[j])
                        goto final_error;

                    chk = _abcdk_sdp_media_sprop_decode(ctx,ctx->fmtp_param[j]->pstrs[0]);
                    if(chk != 0)
                        goto final_error;

                    /**/
                    j++;
                }
            }
        }
        else if (abcdk_strncmp(a_p->obj->pstrs[2], "control:", 8, 0) == 0)
        {
            if (ctx->control)
                goto final_error;

            p_next = a_p->obj->pstrs[2]+8;
            p = abcdk_strtok(&p_next, ";");
            if (!p)
                goto final_error;

            ctx->control = abcdk_object_copyfrom(p,p_next - p);
            if (!ctx->control)
                goto final_error;
        }

        a_p = abcdk_tree_sibling(a_p, 0);
    }

    return ctx;

final_error:

    abcdk_sdp_media_base_free(&ctx);

    return NULL;
}