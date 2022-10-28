/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/rtsp.h"

void _abcdk_rtsp_sdp_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    for (int i = 1; i < alloc->numbers; i++)
        abcdk_heap_free(alloc->pstrs[i]);
}

char *_abcdk_rtsp_sdp_fgetline(FILE *fp, uint8_t delim)
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

void _abcdk_rtsp_sdp_split(abcdk_tree_t *sdp)
{
    FILE *fp = NULL;
    char *p = NULL;

    p = sdp->alloc->pstrs[0];

    fp = fmemopen(p, strlen(p), "r");
    if (!fp)
        goto fianl;

    for (int i = 1; i < 100; i++)
    {
        if (i == 1)
        {
            sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, '=');
            if (!sdp->alloc->pstrs[i])
                break;
        }
        else
        {
            if (*p == 'v' || *p == 's' || *p == 'i' || *p == 'u' || *p == 'e' || *p == 'z' || *p == 'r')
            {
                if (i != 2)
                    break;

                sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, '\n');
                if (!sdp->alloc->pstrs[i])
                    break;
            }
            else
            {
                sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, ' ');
                if (!sdp->alloc->pstrs[i])
                    break;
            }
        }
    }

fianl:

    if (fp)
        fclose(fp);
}

abcdk_tree_t *abcdk_rtsp_sdp_parse(const char *data, size_t size)
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
        abcdk_object_atfree(sub->alloc, _abcdk_rtsp_sdp_destroy_cb, NULL);

        sub->alloc->pstrs[0] = _abcdk_rtsp_sdp_fgetline(fp, '\n');
        if (!sub->alloc->pstrs[0])
            break;

        /*分解字段。*/
        _abcdk_rtsp_sdp_split(sub);

        if (sub->alloc->pstrs[0][0] != 'm')
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

int _abcdk_rtsp_sdp_dump_cb(size_t depth, abcdk_tree_t *node, void *opaque)
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
            if (!node->alloc->pstrs[i])
                break;

            fprintf(fp, "|");
            fprintf(fp, "%s", node->alloc->pstrs[i]);
        }
        fprintf(fp, "|\n");
    }

    return 1;
}

void abcdk_rtsp_sdp_dump(FILE *fp, abcdk_tree_t *sdp)
{
    abcdk_tree_iterator_t it = {0, _abcdk_rtsp_sdp_dump_cb, fp};

    abcdk_tree_scan(sdp, &it);
}

abcdk_tree_t *abcdk_rtsp_sdp_find_media(abcdk_tree_t *sdp, uint8_t fmt)
{
    abcdk_tree_t *p = NULL, *p2 = NULL;

    assert(sdp != NULL && fmt != 0);

    p = abcdk_tree_child(sdp, 1);

    while (p)
    {
        if (p->alloc->pstrs[1][0] == 'm')
        {
            /*遍历媒体格式列表，判断当前节点是否包含需要媒体信息。*/
            for (int i = 5; i < 100; i++)
            {
                if (!p->alloc->pstrs[i])
                    break;

                if (atoi(p->alloc->pstrs[i]) == fmt)
                    return p;
            }
        }

        p = abcdk_tree_sibling(p, 0);
    }

    return NULL;
}

void abcdk_rtsp_sdp_media_base_free(abcdk_rtsp_sdp_media_base_t **ctx)
{
    abcdk_rtsp_sdp_media_base_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_object_unref(&ctx_p->encoder);
    abcdk_object_unref(&ctx_p->extra_vps);
    abcdk_object_unref(&ctx_p->extra_sps);
    abcdk_object_unref(&ctx_p->extra_pps);
    abcdk_object_unref(&ctx_p->extra_sei);
    abcdk_heap_free(ctx_p);
}

abcdk_rtsp_sdp_media_base_t *abcdk_rtsp_sdp_media_base_collect(abcdk_tree_t *sdp, uint8_t fmt)
{
    abcdk_rtsp_sdp_media_base_t *ctx = NULL;
    abcdk_tree_t *a_p = NULL;
    const char *p = NULL, *p_next = NULL, *p_next2 = NULL;
    uint8_t payload, payload2;

    assert(sdp != NULL);

    /*也许传入的是根节点。*/ 
    if(!abcdk_tree_father(sdp))
    {
        sdp = abcdk_rtsp_sdp_find_media(sdp,fmt);
        if(!sdp)
            return NULL;
        
        payload = fmt;
    }
    else if (sdp->alloc->pstrs[1][0] == 'm')
    {
        /*遍历媒体格式列表，判断当前节点是否包含需要媒体信息。*/
        for (int i = 5; i < 100; i++)
        {
            if (!sdp->alloc->pstrs[i])
                return NULL;

            payload = atoi(sdp->alloc->pstrs[i]);
            if (payload == fmt)
                break;
        }
    }
    
    ctx = abcdk_heap_alloc(sizeof(abcdk_rtsp_sdp_media_base_t));
    if (!ctx)
        return NULL;

    /*遍历属性节点。*/
    a_p = abcdk_tree_child(sdp, 1);

    while (a_p)
    {
        if (abcdk_strncmp(a_p->alloc->pstrs[2], "rtpmap:", 7, 0) == 0)
        {
            sscanf(a_p->alloc->pstrs[2], "%*[^:]%*[:]%hhu", &payload2);
            if (payload2 != payload)
            {
                a_p = abcdk_tree_sibling(a_p, 0);
                continue;
            }

            p_next = a_p->alloc->pstrs[3];

            /*拆分编码名称。*/
            p = abcdk_strtok(&p_next, "/");
            if (!p)
                goto final_error;

            ctx->encoder = abcdk_object_alloc2(p_next - p + 1);
            if (!ctx->encoder)
                goto final_error;

            strncpy(ctx->encoder->pstrs[0], p, p_next - p);

            /*拆分时间速率。*/
            p = abcdk_strtok(&p_next, "/");
            if (!p)
                goto final_error;

            sscanf(p, "%u", &ctx->clock_rate);
        }
        else if (abcdk_strncmp(a_p->alloc->pstrs[2], "fmtp:", 5, 0) == 0)
        {
            sscanf(a_p->alloc->pstrs[2], "%*[^:]%*[:]%hhu", &payload2);
            if (payload2 != payload)
            {
                a_p = abcdk_tree_sibling(a_p, 0);
                continue;
            }

            for (int i = 3; i < 100; i++)
            {
                if (!a_p->alloc->pstrs[i])
                    break;

                p_next = a_p->alloc->pstrs[i];

                while (1)
                {
                    p = abcdk_strtok(&p_next, ";");
                    if (!p)
                        break;

                    if (abcdk_strncmp(p, "sprop-parameter-sets=", 21, 0) == 0)
                    {
                        if (ctx->extra_sps || ctx->extra_pps)
                            goto final_error;

                        p_next2 = p + 21;
                        p = abcdk_strtok(&p_next2, ",");
                        if (!p)
                            goto final_error;

                        ctx->extra_sps = abcdk_basecode_decode2(p, p_next2 - p, 64);
                        if (!ctx->extra_sps)
                            goto final_error;

                        p = abcdk_strtok(&p_next2, ",");
                        if (!p)
                            goto final_error;

                        ctx->extra_pps = abcdk_basecode_decode2(p, p_next2 - p, 64);
                        if (!ctx->extra_pps)
                            goto final_error;
                    }
                    else if (abcdk_strncmp(p, "sprop-vps=", 10, 0) == 0)
                    {
                        if (ctx->extra_vps)
                            goto final_error;

                        p_next2 = p + 10;
                        p = abcdk_strtok(&p_next2, ";");
                        if (!p)
                            goto final_error;

                        ctx->extra_vps = abcdk_basecode_decode2(p, p_next2 - p, 64);
                        if (!ctx->extra_vps)
                            goto final_error;
                    }
                    else if (abcdk_strncmp(p, "sprop-sps=", 10, 0) == 0)
                    {
                        if (ctx->extra_sps)
                            goto final_error;

                        p_next2 = p + 10;
                        p = abcdk_strtok(&p_next2, ";");
                        if (!p)
                            goto final_error;

                        ctx->extra_sps = abcdk_basecode_decode2(p, p_next2 - p, 64);
                        if (!ctx->extra_sps)
                            goto final_error;
                    }
                    else if (abcdk_strncmp(p, "sprop-pps=", 10, 0) == 0)
                    {
                        if (ctx->extra_pps)
                            goto final_error;

                        p_next2 = p + 10;
                        p = abcdk_strtok(&p_next2, ";");
                        if (!p)
                            goto final_error;

                        ctx->extra_pps = abcdk_basecode_decode2(p, p_next2 - p, 64);
                        if (!ctx->extra_pps)
                            goto final_error;
                    }
                    else if (abcdk_strncmp(p, "sprop-sei=", 10, 0) == 0)
                    {
                        if (ctx->extra_sei)
                            goto final_error;

                        p_next2 = p + 10;
                        p = abcdk_strtok(&p_next2, ";");
                        if (!p)
                            goto final_error;

                        ctx->extra_sei = abcdk_basecode_decode2(p, p_next2 - p, 64);
                        if (!ctx->extra_sei)
                            goto final_error;
                    }
                }
            }
        }
        else if (abcdk_strncmp(a_p->alloc->pstrs[2], "control:", 8, 0) == 0)
        {
            if (ctx->control)
                goto final_error;

            p_next2 = a_p->alloc->pstrs[2]+8;
            p = abcdk_strtok(&p_next2, ";");
            if (!p)
                goto final_error;

            ctx->control = abcdk_object_alloc2(p_next2 - p + 1);
            if (!ctx->control)
                goto final_error;

            strncpy(ctx->control->pstrs[0], p, p_next2 - p);
        }

        a_p = abcdk_tree_sibling(a_p, 0);
    }

    return ctx;

final_error:

    abcdk_rtsp_sdp_media_base_free(&ctx);

    return NULL;
}