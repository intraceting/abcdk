/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-mp4/atom.h"

void _abcdk_mp4_free_cb(abcdk_allocator_t *alloc, void *opaque)
{
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)alloc->pptrs[0];

    if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HDLR)
    {
        abcdk_mp4_atom_hdlr_t * cont = (abcdk_mp4_atom_hdlr_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->name);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_DREF)
    {
        abcdk_mp4_atom_dref_t * cont = (abcdk_mp4_atom_dref_t *)atom->cont->pptrs[0];
        abcdk_tree_free(&cont->entries);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSD)
    {
        abcdk_mp4_atom_stsd_t * cont = (abcdk_mp4_atom_stsd_t *)atom->cont->pptrs[0];
        abcdk_tree_free(&cont->entries);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_AVC1)
    {
        abcdk_mp4_atom_sample_desc_t * cont = (abcdk_mp4_atom_sample_desc_t *)atom->cont->pptrs[0];
        abcdk_tree_free(&cont->entries);
    }
    else if ((atom->type.u32 == ABCDK_MP4_ATOM_TYPE_GLBL) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_AVCC) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HVCC) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_DVH1))
    {
        abcdk_mp4_atom_glbl_t *cont = (abcdk_mp4_atom_glbl_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->extradata);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STTS)
    {
        abcdk_mp4_atom_stts_t * cont = (abcdk_mp4_atom_stts_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_CTTS)
    {
        abcdk_mp4_atom_ctts_t * cont = (abcdk_mp4_atom_ctts_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSC)
    {
        abcdk_mp4_atom_stsc_t * cont = (abcdk_mp4_atom_stsc_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSZ)
    {
        abcdk_mp4_atom_stsz_t * cont = (abcdk_mp4_atom_stsz_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->tables);
    }
    else if((atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STCO)||
            (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_CO64))
    {
        abcdk_mp4_atom_stco_t * cont = (abcdk_mp4_atom_stco_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSS)
    {
        abcdk_mp4_atom_stss_t * cont = (abcdk_mp4_atom_stss_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_ELST)
    {
        abcdk_mp4_atom_elst_t * cont = (abcdk_mp4_atom_elst_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TRUN)
    {
        abcdk_mp4_atom_trun_t * cont = (abcdk_mp4_atom_trun_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->tables);
    }

    abcdk_allocator_unref(&atom->cont);
}

abcdk_tree_t *abcdk_mp4_alloc()
{
    abcdk_tree_t *node = NULL;

    node = abcdk_tree_alloc3(sizeof(abcdk_mp4_atom_t));
    if(!node)
       return NULL;

    abcdk_allocator_atfree(node->alloc,_abcdk_mp4_free_cb,NULL);

    return node;
}

typedef struct _abcdk_mp4_find_ctx
{
    abcdk_mp4_tag_t type;
    abcdk_tree_t *node;
} abcdk_mp4_find_ctx_t;

int _abcdk_mp4_find_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_find_ctx_t *ctx = NULL;

    if (depth == -1UL)
        return -1;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    ctx = (abcdk_mp4_find_ctx_t *)opaque;
    
    if (atom->type.u32 == ctx->type.u32)
    {
        /*走到这里时，已经找到。*/
        ctx->node = node;
    }
    else if(atom->cont)
    {
        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_DREF)
        {
            abcdk_mp4_atom_dref_t *cont = (abcdk_mp4_atom_dref_t *)atom->cont->pptrs[0];
            ctx->node = abcdk_mp4_find(cont->entries,&atom->type);
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSD)
        {
            abcdk_mp4_atom_stsd_t *cont = (abcdk_mp4_atom_stsd_t *)atom->cont->pptrs[0];
            ctx->node = abcdk_mp4_find(cont->entries,&atom->type);
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_AVC1)
        {
            abcdk_mp4_atom_sample_desc_t *cont = (abcdk_mp4_atom_sample_desc_t *)atom->cont->pptrs[0];
            ctx->node = abcdk_mp4_find(cont->entries,&atom->type);
        }
    }

    /*如果已经找到则中止查找，否则继续找。*/
    return (ctx->node ? -1 : 1);
}

abcdk_tree_t *abcdk_mp4_find(abcdk_tree_t *root, abcdk_mp4_tag_t *type)
{
    abcdk_mp4_find_ctx_t ctx;
    abcdk_tree_t *atom = NULL;

    assert(root != NULL && type != NULL);

    ctx.type = *type;
    ctx.node = NULL;

    abcdk_tree_iterator_t it = {0, _abcdk_mp4_find_cb, &ctx};
    abcdk_tree_scan(root, &it);

    return ctx.node;
}

abcdk_tree_t *abcdk_mp4_find2(abcdk_tree_t *root,uint32_t type)
{
    abcdk_mp4_tag_t tag = {0};

    assert(root != NULL && type != 0);

    tag.u32 = type;

    return abcdk_mp4_find(root,&tag);
}