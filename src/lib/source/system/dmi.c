/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/system/dmi.h"

int _abcdk_dmi_hash_dump_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    /*已经结束。*/
    if(depth == SIZE_MAX)
        return -1;

    if(depth == 0)
        abcdk_tree_fprintf(stderr,depth,node,"%s\n",__FUNCTION__);
    else 
        abcdk_tree_fprintf(stderr,depth,node,"%s\n",node->obj->pstrs[0]);

    return 1;
}

int _abcdk_dmi_hash_compare_cb(const abcdk_tree_t *node1, const abcdk_tree_t *node2, void *opaque)
{
    return abcdk_strcmp(node1->obj->pstrs[0],node2->obj->pstrs[0],1);
}

const uint8_t *abcdk_dmi_hash(uint8_t uuid[16], uint32_t flag, const char *stuff)
{
    abcdk_tree_t *sn_vec = NULL,*mmc_vec = NULL,*scsi_vec = NULL, *p = NULL,*p2 = NULL;
    abcdk_ifaddrs_t ifaddr_vec[100] = {0};
    int ifaddr_count = 0;
    abcdk_md5_t *md5_ctx = NULL;
    int chk = -1;

    assert(uuid != NULL && flag != 0);

    sn_vec = abcdk_tree_alloc3(1);
    if(!sn_vec)
        goto final_error;

    mmc_vec = abcdk_tree_alloc3(1);
    if(!mmc_vec)
        goto final_error;

    scsi_vec = abcdk_tree_alloc3(1);
    if(!scsi_vec)
        goto final_error;

    md5_ctx = abcdk_md5_create();
    if(!md5_ctx)
        goto final_error;

    if (ABCDK_DMI_HASH_USE_DEVICE_MAC & flag)
        ifaddr_count = abcdk_ifname_fetch(ifaddr_vec, 100, 1, 1);

    if (ABCDK_DMI_HASH_USE_DEVICE_MMC & flag)
        abcdk_mmc_list(mmc_vec);
        
    if (ABCDK_DMI_HASH_USE_DEVICE_SCSI & flag)
        abcdk_scsi_list(scsi_vec);

    for(int i = 0;i<ifaddr_count;i++)
    {
        char mac[20] = {0};

        abcdk_mac_fetch(ifaddr_vec[i].name,mac);
        if(!mac[0])
            continue;

        p = abcdk_tree_alloc4(mac,strlen(mac));
        if(!p)
            goto final_error;

        abcdk_tree_insert2(sn_vec, p, 0);
    }

    p2 = abcdk_tree_child(mmc_vec, 1);
    while (p2)
    {
        abcdk_mmc_info_t *dev_p = (abcdk_mmc_info_t *)p2->obj->pptrs[0];

        if (dev_p->cid[0])
        {
            p = abcdk_tree_alloc4(dev_p->cid,strlen(dev_p->cid));
            if (!p)
                goto final_error;

            abcdk_tree_insert2(sn_vec, p, 0);
        }

        p2 = abcdk_tree_sibling(p2, 0);
    }

    p2 = abcdk_tree_child(scsi_vec, 1);
    while (p2)
    {
        abcdk_scsi_info_t *dev_p = (abcdk_scsi_info_t *)p2->obj->pptrs[0];

        if (dev_p->serial[0])
        {
            p = abcdk_tree_alloc4(dev_p->serial,strlen(dev_p->serial));
            if (!p)
                goto final_error;

            abcdk_tree_insert2(sn_vec, p, 0);
        }

        p2 = abcdk_tree_sibling(p2, 0);
    }

    abcdk_tree_iterator_t it = {0,NULL,_abcdk_dmi_hash_dump_cb,_abcdk_dmi_hash_compare_cb};

  //  abcdk_tree_scan(sn_vec,&it);
   // goto final_error;
    abcdk_tree_sort(sn_vec,&it,1);
  //  goto final_error;
    abcdk_tree_distinct(sn_vec,&it);
   // goto final_error;
   // abcdk_tree_scan(sn_vec,&it);

    /*添加自定义干扰项。*/
    if (ABCDK_DMI_HASH_USE_STUFF & flag)
    {
        assert(stuff != NULL && *stuff != '\0');

        p = abcdk_tree_alloc4(stuff, strlen(stuff));
        if (!p)
            goto final_error;

        abcdk_tree_insert2(sn_vec, p, 0);
    }

    abcdk_tree_sort(sn_vec,&it,1);
   // abcdk_tree_scan(sn_vec,&it);

    p2 = abcdk_tree_child(sn_vec, 1);
    while (p2)
    {
        abcdk_md5_update(md5_ctx,p2->obj->pstrs[0],strlen(p2->obj->pstrs[0]));

        p2 = abcdk_tree_sibling(p2, 0);
    }
    
    abcdk_md5_final(md5_ctx,uuid);
    chk = 0;

    goto final;

final_error:

    chk = -1;
    
final:

    abcdk_tree_free(&sn_vec);
    abcdk_tree_free(&mmc_vec);
    abcdk_tree_free(&scsi_vec);
    abcdk_md5_destroy(&md5_ctx);

    return (chk==0?uuid:NULL);
}
