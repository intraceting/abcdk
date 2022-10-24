/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/rtsp.h"

void _abcdk_rtsp_sdp_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    for(int i = 1;i<alloc->numbers;i++)
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
        if(line[rlen - 1] == delim)
            line[rlen - 1] = '\0';

        /*去掉字符串两端所有空白字符。 */
        abcdk_strtrim(line, isspace, 2);

        /*跳过空行。*/
        if(*line == '\0')
            goto next_line;

        p = abcdk_heap_clone(line,strlen(line));
        if(!p)
            goto final;
    }

final:

    if(line)
        free(line);

    return p;
}

void _abcdk_rtsp_sdp_split(abcdk_tree_t *sdp)
{
    FILE *fp = NULL;
    char *p = NULL;

    p = sdp->alloc->pstrs[0];

    fp = fmemopen(p,strlen(p),"r");
    if(!fp)
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
            if (*p == 'v' || *p == 's' || *p == 'i' || *p == 'u' || *p == 'e' || *p == 'z' || *p == 'k' || *p == 'r')
            {
                if (i != 2)
                    break;

                sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, '\n');
                if (!sdp->alloc->pstrs[i])
                    break;
            }
            else if (*p == 'o')
            {
                if (i < 2 || i > 5)
                    break;

                sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, ' ');
                if (!sdp->alloc->pstrs[i])
                    break;
            }
            else if (*p == 'c')
            {
                if (i < 2 || i > 4)
                    break;

                sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, ' ');
                if (!sdp->alloc->pstrs[i])
                    break;                
            }
            else if (*p == 't')
            {
                if (i < 2 || i > 3)
                    break;

                sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, ' ');
                if (!sdp->alloc->pstrs[i])
                    break;                
            }
            else if (*p == 'a' || *p == 'b')
            {
                if (i == 2)
                {
                    sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, ':');
                    if (!sdp->alloc->pstrs[i])
                        break;
                }
                else
                {
                    if (i == 3)
                    {
                        sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, ' ');
                        if (!sdp->alloc->pstrs[i])
                            break;
                    }
                    else 
                    {
                        sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, ';');
                        if (!sdp->alloc->pstrs[i])
                            break;
                    }
                }
            }
            else if (*p == 'm')
            {
                if (i < 2)
                    break;

                sdp->alloc->pstrs[i] = _abcdk_rtsp_sdp_fgetline(fp, ' ');
                if (!sdp->alloc->pstrs[i])
                    break;                
            }
        }
    }

fianl:

    if(fp)
        fclose(fp);
}

abcdk_tree_t *abcdk_rtsp_sdp_parse(const char *data, size_t size)
{
    abcdk_tree_t *sdp = NULL,*sub = NULL,*sdp_p = NULL;
    FILE *fp = NULL;

    assert(data != NULL && size > 0);

    fp = fmemopen((void*)data,size,"r");
    if(!fp)
        goto fianl_error;
    
    sdp = abcdk_tree_alloc3(1);
    if(!sdp)
        goto fianl_error;

    sdp_p = sdp;

    while (1)
    {
        sub = abcdk_tree_alloc2(NULL, 100, 0);
        if(!sub)
            goto fianl_error;
        
        /*注册清理函数。*/
        abcdk_object_atfree(sub->alloc,_abcdk_rtsp_sdp_destroy_cb,NULL);

        sub->alloc->pstrs[0] = _abcdk_rtsp_sdp_fgetline(fp, '\n');
        if (!sub->alloc->pstrs[0])
            break;

        /*分解字段。*/
        _abcdk_rtsp_sdp_split(sub);
        
        if(sub->alloc->pstrs[0][0] != 'm')
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

    if(fp)
        fclose(fp);
    
    return sdp;
}

int _abcdk_rtsp_sdp_dump_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    FILE *fp = (FILE*)opaque;

    if(depth == 0)
    {
        abcdk_tree_fprintf(fp, depth, node,"SDP\n");
    }
    else if(depth == SIZE_MAX)
    {
        return -1;
    }
    else 
    {
        abcdk_tree_fprintf(fp, depth, node, "");
        for (int i = 1; i < 100; i++)
        {
            if(!node->alloc->pstrs[i])
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
    abcdk_tree_iterator_t it = {0,_abcdk_rtsp_sdp_dump_cb,fp};

    abcdk_tree_scan(sdp,&it);
}

abcdk_tree_t *abcdk_rtsp_sdp_find_media_info(abcdk_tree_t *sdp, uint8_t fmt, const char *type,const char *sub)
{
    abcdk_tree_t *p = NULL, *p2 = NULL;

    assert(sdp != NULL && fmt != 0);

    p = abcdk_tree_child(sdp,1);

    while(p)
    {
        if(p->alloc->pstrs[1][0] != 'm' || atoi(p->alloc->pstrs[5]) != fmt)
        {
            p = abcdk_tree_sibling(p,0);
        }
        else 
        {
            /*如果不需要查找属性，则直接返回。*/
            if (!type)
                return p;

            p2 = abcdk_tree_child(p,1);

            while(p2)
            {
                if(p2->alloc->pstrs[1][0] != *type)
                {
                    p2 = abcdk_tree_sibling(p2,0);
                }
                else 
                {
                    /*如果不需要查找子属性，则直接返回。*/
                    if(!sub)
                        return p2;

                    if(abcdk_strcmp(p2->alloc->pstrs[2],sub,1)!=0)
                    {
                        p2 = abcdk_tree_sibling(p2,0);
                    }
                    else 
                    {
                        return p2;
                    }
                }
            }
        }
    }

    return NULL;
}