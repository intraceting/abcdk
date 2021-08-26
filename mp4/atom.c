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

    if(!atom->have_data)
        return;

    if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_FTYP)
    {
        abcdk_mp4_atom_ftyp_t * data = (abcdk_mp4_atom_ftyp_t *)&atom->data;
        abcdk_allocator_unref(&data->compat);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HDLR)
    {
        abcdk_mp4_atom_hdlr_t * data = (abcdk_mp4_atom_hdlr_t *)&atom->data;
        abcdk_allocator_unref(&data->name);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STTS)
    {
        abcdk_mp4_atom_stts_t * data = (abcdk_mp4_atom_stts_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_CTTS)
    {
        abcdk_mp4_atom_ctts_t * data = (abcdk_mp4_atom_ctts_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSC)
    {
        abcdk_mp4_atom_stsc_t * data = (abcdk_mp4_atom_stsc_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSZ)
    {
        abcdk_mp4_atom_stsz_t * data = (abcdk_mp4_atom_stsz_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if((atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STCO)||
            (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_CO64))
    {
        abcdk_mp4_atom_stco_t * data = (abcdk_mp4_atom_stco_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSS)
    {
        abcdk_mp4_atom_stss_t * data = (abcdk_mp4_atom_stss_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_ELST)
    {
        abcdk_mp4_atom_elst_t * data = (abcdk_mp4_atom_elst_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TRUN)
    {
        abcdk_mp4_atom_trun_t * data = (abcdk_mp4_atom_trun_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TFRA)
    {
        abcdk_mp4_atom_tfra_t * data = (abcdk_mp4_atom_tfra_t *)&atom->data;
        abcdk_heap_free(data->tables);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MP4A)
    {
        abcdk_mp4_atom_sample_desc_t * data = (abcdk_mp4_atom_sample_desc_t *)&atom->data;
        if (data->detail.sound.version == 2)
            abcdk_allocator_unref(&data->detail.sound.v2.extension);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STPP)
    {
        abcdk_mp4_atom_sample_desc_t * data = (abcdk_mp4_atom_sample_desc_t *)&atom->data;
        abcdk_allocator_unref(&data->detail.subtitle.extension);
    }
    else if ((atom->type.u32 == ABCDK_MP4_ATOM_TYPE_GLBL) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_AVCC) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HVCC) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_DVH1) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_PRIV) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_ALIS) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_UUID))
    {
        abcdk_mp4_atom_glbl_t *data = (abcdk_mp4_atom_glbl_t *)&atom->data;
        abcdk_allocator_unref(&data->extradata);
    }
    else if ((atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TKHD) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_AVC1) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_VMHD) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TFDT) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TFHD) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_FREE) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_SKIP) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_WIDE) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MDAT) )
    {
        /*忽略。*/
    }
    else
    {
        abcdk_mp4_atom_unknown_t *data = (abcdk_mp4_atom_unknown_t *)&atom->data;
        abcdk_allocator_unref(&data->rawbytes);
    }

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
    size_t hsize = 0, dsize = 0;

    if (depth == -1UL)
        return -1;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    ctx = (abcdk_mp4_dump_ctx_t *)opaque;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    abcdk_tree_fprintf(ctx->fd, depth, node, "[%c%c%c%c] size=%lu+%lu offset=%lu",
                       atom->type.u8[0], atom->type.u8[1], atom->type.u8[2], atom->type.u8[3],
                       hsize, dsize, atom->off_head);

    fprintf(ctx->fd,"\n");

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

int abcdk_mp4_stsc_tell(abcdk_mp4_atom_stsc_t *stsc, uint32_t sample, uint32_t *chunk, uint32_t *offset, uint32_t *id)
{
    uint32_t chunk_cursor = 0;
    uint32_t sample_offset = 0;

    assert(sample > 0 && chunk != NULL && offset != NULL && id != NULL);

    for (size_t i = 0; i < stsc->numbers; i++)
    {
       chunk_cursor += sample/stsc->tables[i].samples_perchunk;
    }
}