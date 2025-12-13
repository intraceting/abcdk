/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/system/mediumx.h"


typedef struct _abcdk_mediumx_element_status_format_param
{
    int fmt;
    FILE *out;

    int head_out;

    /*设备列表.*/
    abcdk_tree_t *devs;

}abcdk_mediumx_element_status_format_param_t;

const char *_abcdk_mediumx_element_status_format_find_devname(abcdk_mediumx_element_status_format_param_t *param, uint8_t type, const char *sn)
{
    abcdk_tree_t *node_p = NULL;
    abcdk_scsi_info_t *dev_p = NULL;

    node_p = abcdk_tree_child(param->devs, 1);
    while (node_p)
    {
        dev_p = (abcdk_scsi_info_t *)node_p->obj->pptrs[0];

        if (dev_p->serial[0] != '\0')
        {
            if (dev_p->type == TYPE_TAPE && type == ABCDK_MEDIUMX_ELEMENT_DXFER)
            {
                if (abcdk_strcmp(dev_p->serial, sn, 1) == 0)
                    return dev_p->devname;
            }

            if (dev_p->type == TYPE_MEDIUM_CHANGER && type == ABCDK_MEDIUMX_ELEMENT_CHANGER)
            {
                if (abcdk_strcmp(dev_p->serial, sn, 1) == 0)
                    return dev_p->generic;
            }
        }

        node_p = abcdk_tree_sibling(node_p, 0);
    }

    return sn;
}

int _abcdk_mediumx_element_status_format_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_mediumx_element_status_format_param_t *param_p = (abcdk_mediumx_element_status_format_param_t*)opaque;
    uint16_t addr;
    uint8_t type;
    uint8_t full;
    const char *dvcid;
    const char *barcode;

    if (depth == 0)
    {
        if(param_p->fmt == 2)
        {
            fprintf(param_p->out,"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
            fprintf(param_p->out,"\t<elements>\n");
        }
        else if(param_p->fmt == 3)
        {
            fprintf(param_p->out,"{\n");
            fprintf(param_p->out,"\t\"elements\":[\n");
        }
        else if(param_p->fmt == 1)
        {
            abcdk_tree_fprintf(param_p->out,depth, node, "elements\n");
        }
    }
    else if (depth == SIZE_MAX)
    {
        if(param_p->fmt == 2)
        {
            fprintf(param_p->out,"\t</elements>\n");
            fprintf(param_p->out,"</library>\n");
        }
        else if(param_p->fmt == 3)
        {
            fprintf(param_p->out,"\t]\n");
            fprintf(param_p->out,"}\n");
        }
    }
    else
    {
        addr = ABCDK_PTR2U16(node->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_ADDR], 0);
        type = ABCDK_PTR2U8(node->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_TYPE], 0);
        full = ABCDK_PTR2U8(node->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_ISFULL], 0);
        dvcid = _abcdk_mediumx_element_status_format_find_devname(param_p,type,(char*)node->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_DVCID]);
        barcode = (char*)node->obj->pptrs[ABCDK_MEDIUMX_ELEMENT_BARCODE];

        if (param_p->fmt == 2)
        {
            fprintf(param_p->out, "\t\t<element address=\"%hu\" type=\"%hhu\" full=\"%hhu\" dvcid=\"%s\" >%s</element>\n",
                    addr,type,full,dvcid,barcode);
        }
        else if(param_p->fmt == 3)
        {
            fprintf(param_p->out, "\t\t{\n");
            fprintf(param_p->out, "\t\t\t\"address\":\"%hu\",\n",addr);
            fprintf(param_p->out, "\t\t\t\"type\":\"%hhu\",\n",type);
            fprintf(param_p->out, "\t\t\t\"full\":\"%hhu\",\n",full);
            fprintf(param_p->out, "\t\t\t\"barcode\":\"%s\"\n",barcode);
            fprintf(param_p->out, "\t\t\t\"dvcid\":\"%s\",\n",dvcid);
            fprintf(param_p->out, "\t\t}");
            fprintf(param_p->out, "%s\n",(abcdk_tree_sibling(node,0)?",":""));
             
        }
        else if(param_p->fmt == 1)
        {
            if (param_p->head_out++ <= 0)
                abcdk_tree_fprintf(param_p->out, depth, node, "%-6s\t|%-2s\t|%-2s\t|%-7s\t|%-10s\t|\n",
                                   "address", "type", "full", "barcode", "dvcid");

            abcdk_tree_fprintf(param_p->out, depth, node, "%-6hu\t|%-2hhu\t|%-2hhu\t|%-7s\t|%-10s\t|\n",
                               addr,type,full,barcode,dvcid);
        }
    }

    return 1;//next.
}

int abcdk_mediumx_element_status_format(abcdk_tree_t *list,int fmt, FILE* out)
{
    abcdk_mediumx_element_status_format_param_t param = {0};

    assert(list != NULL && fmt != 0 && out != NULL);
    assert(fmt == 1 || fmt == 2 || fmt == 3);

    param.fmt = fmt;
    param.out = out;

    param.head_out = 0;//must be 0.

    /*枚举设备.*/
    abcdk_scsi_fetch2(&param.devs);

    abcdk_tree_iterator_t it = {0, &param, _abcdk_mediumx_element_status_format_cb};
    abcdk_tree_scan(list, &it);

    abcdk_tree_free(&param.devs);
}
