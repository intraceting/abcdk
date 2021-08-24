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

    if(!atom->cont)
        goto final;

    if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HDLR)
    {
        abcdk_mp4_atom_hdlr_t * cont = (abcdk_mp4_atom_hdlr_t *)atom->cont->pptrs[0];
        abcdk_allocator_unref(&cont->name);
    }
    else if ((atom->type.u32 == ABCDK_MP4_ATOM_TYPE_GLBL) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_AVCC) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HVCC) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_DVH1) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_PRIV) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_ALIS))
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

final:

    abcdk_allocator_unref(&atom->cont);
    abcdk_tree_free(&atom->entries);
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
    size_t index;
    int recursive;

    size_t index2;
    abcdk_tree_t *ret;
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
        ctx->ret = node;
    }

    if (ctx->ret)
    {
        if (ctx->index == ++ctx->index2)
            return -1;
        else 
            ctx->ret = NULL;// 不是想找到，继续找。
    }

    /* 当前节点深度大于或等于1时，要考虑是否递归查找。*/
    return ((depth >= 1) ? (ctx->recursive ? 1 : 0) : 1);
}

abcdk_tree_t *abcdk_mp4_find(abcdk_tree_t *root, abcdk_mp4_tag_t *type,size_t index,int recursive)
{
    abcdk_mp4_find_ctx_t ctx;
    abcdk_tree_t *atom = NULL;

    assert(root != NULL && type != NULL && index > 0);

    ctx.type = *type;
    ctx.index = index;
    ctx.recursive = recursive;
    ctx.index2 = 0;
    ctx.ret = NULL;

    abcdk_tree_iterator_t it = {0, _abcdk_mp4_find_cb, &ctx};
    abcdk_tree_scan(root, &it);

    return ctx.ret;
}

abcdk_tree_t *abcdk_mp4_find2(abcdk_tree_t *root,uint32_t type,size_t index,int recursive)
{
    abcdk_mp4_tag_t tag = {0};

    assert(root != NULL && type != 0 && index > 0);

    tag.u32 = type;

    return abcdk_mp4_find(root,&tag,index,recursive);
}

typedef struct _abcdk_mp4_dump_ctx
{
    FILE *fd;
} abcdk_mp4_dump_ctx_t;


int _abcdk_mp4_dump_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_dump_ctx_t *ctx = NULL;
    abcdk_tree_t *sub_node = NULL;
    abcdk_mp4_dump_ctx_t sub_ctx;

    if (depth == -1UL)
        return -1;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    ctx = (abcdk_mp4_dump_ctx_t *)opaque;

    abcdk_tree_fprintf(ctx->fd, depth, node, "[%c%c%c%c] size=%lu offset=%lu\n",
                       atom->type.u8[0], atom->type.u8[1], atom->type.u8[2], atom->type.u8[3],
                       atom->size, atom->off_head);


    return 1;
}

void abcdk_mp4_dump(FILE *fd, abcdk_tree_t *root)
{
    abcdk_mp4_dump_ctx_t ctx;

    assert(root != NULL);

    ctx.fd = fd;

    abcdk_tree_iterator_t it = {0, _abcdk_mp4_dump_cb, &ctx};
    abcdk_tree_scan(root, &it);
}