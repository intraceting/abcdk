/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/mp4/atom.h"


void _abcdk_mp4_free_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)alloc->pptrs[0];

    if(!atom->have_data)
        return;

    if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_FTYP)
    {
        abcdk_mp4_atom_ftyp_t * data = (abcdk_mp4_atom_ftyp_t *)&atom->data;
        abcdk_object_unref(&data->compat);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HDLR)
    {
        abcdk_mp4_atom_hdlr_t * data = (abcdk_mp4_atom_hdlr_t *)&atom->data;
        abcdk_object_unref(&data->name);
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
            abcdk_object_unref(&data->detail.sound.v2.extension);
    }
    else if(atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STPP)
    {
        abcdk_mp4_atom_sample_desc_t * data = (abcdk_mp4_atom_sample_desc_t *)&atom->data;
        abcdk_object_unref(&data->detail.subtitle.extension);
    }
    else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_AVCC)
    {
        abcdk_mp4_atom_avcc_t *data = (abcdk_mp4_atom_avcc_t *)&atom->data;
        abcdk_object_unref(&data->extradata);
    }
    else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HVCC)
    {
        abcdk_mp4_atom_hvcc_t *data = (abcdk_mp4_atom_hvcc_t *)&atom->data;
        abcdk_object_unref(&data->extradata);
    }
    else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_ESDS)
    {
        abcdk_mp4_atom_esds_t *data = (abcdk_mp4_atom_esds_t *)&atom->data;
        abcdk_object_unref(&data->dec_sp_info.extradata);
    }
    else if ((atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TKHD) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MDHD) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_AVC1) ||
             (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HEV1) ||
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
        abcdk_object_unref(&data->rawbytes);
    }

}

abcdk_tree_t *abcdk_mp4_alloc()
{
    abcdk_tree_t *node = NULL;

    node = abcdk_tree_alloc3(sizeof(abcdk_mp4_atom_t));
    if(!node)
       return NULL;

    abcdk_object_atfree(node->obj,_abcdk_mp4_free_cb,NULL);

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

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
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

    abcdk_tree_iterator_t it = {0, &ctx, _abcdk_mp4_find_cb};
    abcdk_tree_scan(root, &it);

    return ctx.ret;
}

abcdk_tree_t *abcdk_mp4_find2(abcdk_tree_t *root,uint32_t type,size_t index,int recursive)
{
    abcdk_mp4_tag_t tag;

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

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    ctx = (abcdk_mp4_dump_ctx_t *)opaque;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    abcdk_tree_fprintf(ctx->fd, depth, node, "[%c%c%c%c] size=%zu+%zu offset=%llu",
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

    abcdk_tree_iterator_t it = {0, &ctx, _abcdk_mp4_dump_cb};
    abcdk_tree_scan(root, &it);
}

int abcdk_mp4_stsc_tell(abcdk_mp4_atom_stsc_t *stsc, uint32_t sample, uint32_t *chunk, uint32_t *offset, uint32_t *id)
{
    uint32_t sample_count = 0;

    assert(sample > 0 && chunk != NULL && offset != NULL && id != NULL);

    /* 遍历采样表，找到即返回。*/

    for (size_t i = 0; i < stsc->numbers; i++)
    {
        if (stsc->tables[i].samples_perchunk <= 0)
            goto final_error;

        /*当存在多种不同的chunk时，要检测sample是否在后面的chunk中。*/
        if (i + 1 < stsc->numbers)
        {
            if(stsc->tables[i + 1].samples_perchunk<=0)
                goto final_error;

            sample_count = (stsc->tables[i + 1].first_chunk - stsc->tables[i].first_chunk) * stsc->tables[i].samples_perchunk;
            if (sample_count < sample)
            {
                /* 每一种chunk存储的sample数量可能不相等，这里每次减去N*M个。*/
                sample -= sample_count;
            }
            else
            {
                *id = stsc->tables[i].sample_desc_id;
                if (sample % stsc->tables[i].samples_perchunk)
                {
                    *chunk = stsc->tables[i].first_chunk + (sample / stsc->tables[i].samples_perchunk); 
                    *offset = sample % stsc->tables[i].samples_perchunk;
                }
                else
                {
                    *chunk = stsc->tables[i].first_chunk + (sample / stsc->tables[i].samples_perchunk) - 1;// 以1为基值。
                    *offset = stsc->tables[i].samples_perchunk; 
                }

                return 0;
            }
        }
        else
        {
            *id = stsc->tables[i].sample_desc_id;
            if (sample % stsc->tables[i].samples_perchunk)
            {
                *chunk = stsc->tables[i].first_chunk + (sample / stsc->tables[i].samples_perchunk);
                *offset = sample % stsc->tables[i].samples_perchunk;
            }
            else
            {
                *chunk = stsc->tables[i].first_chunk + (sample / stsc->tables[i].samples_perchunk) - 1;// 以1为基值。
                *offset = stsc->tables[i].samples_perchunk;
            }

            return 0;
        }
    }

final_error:

    *chunk = -1U;
    *offset = -1U;
    *id = -1U;

    return -1;
}

int abcdk_mp4_stsz_tell(abcdk_mp4_atom_stsz_t *stsz, uint32_t off_chunk, uint32_t sample, uint32_t *offset, uint32_t *size)
{
    assert(stsz != NULL && off_chunk > 0 && sample > 0 && offset != NULL && size != NULL);

    assert(off_chunk <= sample);

    if (stsz->numbers < sample)
        goto final_error;

    /*sample在chunk内连续存储，连续累加即可找到。*/

    for (size_t i = (sample - off_chunk + 1); i <= stsz->numbers && i < sample; i++)
    {
        *offset += stsz->tables[i - 1].size;
    }

    *size = stsz->tables[sample - 1].size;

    return 0;

final_error:

    *offset = -1U;
    *size = 0;

    return -1;
}

int abcdk_mp4_stts_tell(abcdk_mp4_atom_stts_t *stts, uint32_t sample, uint64_t *dts, uint32_t *duration)
{
    uint32_t sample_start = 0;
    uint64_t dts_start = 0;

    assert(stts != NULL && sample > 0 && dts != NULL && duration != NULL);
    
    /* 遍历采样表，找到即返回。*/

    for (size_t i = 0; i < stts->numbers; i++)
    {
        /* 表中记录的是一组帧的帧差，这里要找到最接近的，才能计算出真的DTS。*/
        if (sample <= sample_start + stts->tables[i].sample_count)
        {
            *dts = dts_start + (uint64_t)(sample - sample_start) * (uint64_t)stts->tables[i].sample_duration;
            *duration = stts->tables[i].sample_duration;

            return 0;
        }

        /* 累加帧差，作为下一组帧的DTS开始。*/
        sample_start += stts->tables[i].sample_count;
        dts_start += (uint64_t)stts->tables[i].sample_count * (uint64_t)stts->tables[i].sample_duration;
    }

    return -1;
}

int abcdk_mp4_ctts_tell(abcdk_mp4_atom_ctts_t *ctts,uint32_t sample,  int32_t* offset)
{
    uint32_t sample_start = 0;

    assert(ctts != NULL && sample > 0 && offset != NULL);

    /* 没有B帧时，表可能是空的。*/
    if (ctts->numbers <= 0)
    {
        *offset = 0;
        return 0;
    }

    /* 遍历采样表，找到即返回。*/

    for (size_t i = 0; i < ctts->numbers; i++)
    {
        /* 表中记录的是一组帧的帧差，这里要找到最接近的，才能计算出真的CTS。*/
        if (sample <= sample_start + ctts->tables[i].sample_count)
        {
            *offset = ctts->tables[i].composition_offset;

            return 0;
        }

        /* 累加帧差，作为下一组帧的CTS开始。*/
        sample_start += ctts->tables[i].sample_count;
    }

    return -1;
}

int abcdk_mp4_stss_tell(abcdk_mp4_atom_stss_t *stss, uint32_t sample)
{
    assert(stss != NULL && sample > 0);

    /*如果表是空的，全部是关键帧。*/
    if (stss->numbers <= 0)
        return 0;

    for (size_t i = 0; i < stss->numbers; i++)
    {
        if (stss->tables[i].sync == sample)
            return 0;
    }

    return -1;
}