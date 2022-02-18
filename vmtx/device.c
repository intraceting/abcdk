/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "device.h"

int _abcdk_vmtx_dev_find(abcdk_tree_t *list, abcdk_tree_t *node)
{
    abcdk_tree_t *p;
    abcdk_scsi_info_t *dev_p, *dev_q;

    /*链表为空，直接返回“未找到”。*/
    if (!list)
        return 0;

    dev_q = (abcdk_scsi_info_t *)node->alloc->pptrs[0];

    p = abcdk_tree_child(list, 1);
    while (p)
    {
        dev_p = (abcdk_scsi_info_t *)p->alloc->pptrs[0];

        if (abcdk_strcmp(dev_p->serial, dev_q->serial,1) == 0 &&
            abcdk_strcmp(dev_p->devname, dev_q->devname,1) == 0 &&
            abcdk_strcmp(dev_p->generic, dev_q->generic,1) == 0)
        {
            return 1;
        }

        p = abcdk_tree_sibling(p, 0);
    }

    return 0;
}

void _abcdk_vmtx_dev_diff(abcdk_tree_t *old_list,abcdk_tree_t *new_list,abcdk_tree_t **diff,int add)
{
    abcdk_tree_t *p;
    abcdk_tree_t *tmp;
    abcdk_tree_t *diff_p;

    abcdk_tree_free(diff);
    *diff = abcdk_tree_alloc3(1);
    if (!*diff)
        return;

    diff_p = *diff;

    /*
     * add == 1 : 从旧的中查找新的。
     * del == 0 ; 从新的中查找旧的。
    */
   
    p = abcdk_tree_child((add ? new_list : old_list), 1);
    while (p)
    {
        if (!_abcdk_vmtx_dev_find((add ? old_list : new_list), p))
        {
            tmp = abcdk_tree_alloc(abcdk_allocator_refer(p->alloc));//增加引用计数。
            if (!tmp)
                return;

            abcdk_tree_insert2(diff_p, tmp, 0);
        }

        p = abcdk_tree_sibling(p, 0);
    }

    /*如果未发生变化，删除差异链表。*/
    if(!abcdk_tree_child(diff_p,1))
        abcdk_tree_free(diff);

    return;
}

void _abcdk_vmtx_dev_check_ok(abcdk_tree_t *list)
{
    abcdk_tree_t *p,*tmp;
    abcdk_scsi_info_t *dev_p;

    p = abcdk_tree_child(list, 1);
    while (p)
    {
        dev_p = (abcdk_scsi_info_t *)p->alloc->pptrs[0];

        if (dev_p->vendor[0] == '\0' || dev_p->model[0] == '\0' ||
            dev_p->revision[0] == '\0' || dev_p->serial[0] == '\0' ||
            dev_p->devname[0] == '\0' || dev_p->generic[0] == '\0')
        {
            /*下一个节点。*/
            p = abcdk_tree_sibling(tmp = p, 0);

            /*删除不完成的节点。*/
            abcdk_tree_unlink(tmp);
            abcdk_tree_free(&tmp);
        }
        else 
        {
            /*下一个节点。*/
            p = abcdk_tree_sibling(p, 0);
        }
    }
}

void abcdk_vmtx_dev_watch(abcdk_tree_t **snapshot, abcdk_tree_t **add, abcdk_tree_t **del)
{
    abcdk_tree_t *tmp = NULL;

    assert(snapshot != NULL);

    tmp = abcdk_tree_alloc3(1);
    if (!tmp)
        return;

    abcdk_scsi_list(tmp);
    _abcdk_vmtx_dev_check_ok(tmp);

    if(*snapshot)
    {
        if(add)
            _abcdk_vmtx_dev_diff(*snapshot,tmp,add,1);
        if(del)
            _abcdk_vmtx_dev_diff(*snapshot,tmp,del,0);
    }
    else
    {
        if(add)
            _abcdk_vmtx_dev_diff(NULL,tmp,add,1);
    }
    
    abcdk_tree_free(snapshot);
    _abcdk_vmtx_dev_diff(NULL,tmp,snapshot,1);
    
    abcdk_tree_free(&tmp);
}
