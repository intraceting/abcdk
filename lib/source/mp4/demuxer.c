/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/mp4/demuxer.h"

/*
 * 读取atom头，但不改变文件指针位置。
*/
abcdk_tree_t *abcdk_mp4_read_header(int fd)
{
    uint64_t fsize = 0;
    uint32_t size32 = 0;
    uint64_t size64 = 0;
    abcdk_tree_t *node = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    int chk;

    assert(fd >= 0);

    chk = abcdk_mp4_size(fd, &fsize);
    if (chk != 0)
        goto final_error;

    node = abcdk_mp4_alloc();
    if (!node)
        return NULL;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];

    /* 保存数据偏移量。*/
    atom->off_head = lseek(fd, 0, SEEK_CUR);

    chk = abcdk_mp4_read_u32(fd, &size32);
    if (chk != 0)
        goto final_error;

    chk = abcdk_mp4_read(fd, &atom->type.u32, sizeof(atom->type.u32));
    if (chk != 0)
        goto final_error;

    /* 当size32==1时，需要读取扩展字段来确定长度。*/
    if (size32 == 1)
    {
        chk = abcdk_mp4_read_u64(fd, &size64);
        if (chk != 0)
            goto final_error;

        atom->size = size64;
        atom->off_data = atom->off_head + 16;
    }
    else if (size32 == 0)
    {
        atom->size = fsize - atom->off_head;
        atom->off_data = atom->off_head + 8;
    }
    else
    {
        atom->size = size32;
        atom->off_data = atom->off_head + 8;
    }

    /*不能超过文件末尾。*/
    if (atom->off_head + atom->size > fsize)
        atom->size = fsize - atom->off_head;

    /* 恢复偏移量。*/
    lseek(fd, atom->off_head, SEEK_SET);

    return node;

final_error:

    abcdk_tree_free(&node);

    return NULL;
}

int abcdk_mp4_read_fullheader(int fd, uint8_t *ver, uint32_t *flags)
{
    assert(fd >= 0 && ver != NULL && flags != NULL);

    uint32_t h = 0;
    if (abcdk_mp4_read_u32(fd, &h))
        return -1;

    *ver = ((h >> 24) & 0x000000FF);
    *flags = (h & 0x00FFFFFF);

    return 0;
}

/*在这里声明一下，实现在下面。*/
int _abcdk_mp4_read_probe(abcdk_tree_t *root, int fd, abcdk_mp4_tag_t *stop);


int _abcdk_mp4_read_ftyp(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_ftyp_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_ftyp_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read(fd, &data->major.u32, 4);
    abcdk_mp4_read_u32(fd, &data->minor);

    data->compat = abcdk_object_alloc3(4, (dsize - 8) / 4);
    if (!data->compat)
        goto final_error;

    for (size_t i = 0; i < data->compat->numbers; i++)
        abcdk_mp4_read(fd, data->compat->pptrs[i], 4);

    return 0;

final_error:

    abcdk_object_unref(&data->compat);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_mvhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mvhd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_mvhd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    if (data->version == 0)
    {
        abcdk_mp4_read_u32to64(fd, &data->ctime);
        abcdk_mp4_read_u32to64(fd, &data->mtime);
        abcdk_mp4_read_u32(fd, &data->timescale);
        abcdk_mp4_read_u32to64(fd, &data->duration);
    }
    else
    {
        abcdk_mp4_read_u64(fd, &data->ctime);
        abcdk_mp4_read_u64(fd, &data->mtime);
        abcdk_mp4_read_u32(fd, &data->timescale);
        abcdk_mp4_read_u64(fd, &data->duration);
    }

    abcdk_mp4_read_u32(fd, &data->rate);
    abcdk_mp4_read_u16(fd, &data->volume);
    abcdk_mp4_read(fd, data->reserved, 10);

    for (int i = 0; i < 9; i++)
        abcdk_mp4_read_u32(fd, &data->matrix[i]);

    abcdk_mp4_read(fd, data->predefined, sizeof(data->predefined));
    abcdk_mp4_read_u32(fd, &data->nexttrackid);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_udta(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    
    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd,atom->off_data,SEEK_SET);

    chk = _abcdk_mp4_read_probe(node, fd, NULL);
    if(chk == -1)
        return -1;

    return 0;
}

int _abcdk_mp4_read_tkhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_tkhd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_tkhd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    if (data->version == 0)
    {
        abcdk_mp4_read_u32to64(fd, &data->ctime);
        abcdk_mp4_read_u32to64(fd, &data->mtime);
        abcdk_mp4_read_u32(fd, &data->trackid);
        abcdk_mp4_read_u32(fd, &data->reserved1);
        abcdk_mp4_read_u32to64(fd, &data->duration);
    }
    else
    {
        abcdk_mp4_read_u64(fd, &data->ctime);
        abcdk_mp4_read_u64(fd, &data->mtime);
        abcdk_mp4_read_u32(fd, &data->trackid);
        abcdk_mp4_read_u32(fd, &data->reserved1);
        abcdk_mp4_read_u64(fd, &data->duration);
    }

    abcdk_mp4_read_u64(fd, &data->reserved2);
    abcdk_mp4_read_u16(fd, &data->layer);
    abcdk_mp4_read_u16(fd, &data->alternategroup);
    abcdk_mp4_read_u16(fd, &data->volume);
    abcdk_mp4_read_u16(fd, &data->reserved3);

    for (int i = 0; i < 9; i++)
        abcdk_mp4_read_u32(fd, &data->matrix[i]);

    abcdk_mp4_read_u32(fd, &data->width);
    abcdk_mp4_read_u32(fd, &data->height);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_mdhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mdhd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_mdhd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    if (data->version == 0)
    {
        abcdk_mp4_read_u32to64(fd, &data->ctime);
        abcdk_mp4_read_u32to64(fd, &data->mtime);
        abcdk_mp4_read_u32(fd, &data->timescale);
        abcdk_mp4_read_u32to64(fd, &data->duration);
    }
    else
    {
        abcdk_mp4_read_u64(fd, &data->ctime);
        abcdk_mp4_read_u64(fd, &data->mtime);
        abcdk_mp4_read_u32(fd, &data->timescale);
        abcdk_mp4_read_u64(fd, &data->duration);
    }

    abcdk_mp4_read_u16(fd, &data->language);
    abcdk_mp4_read_u16(fd, &data->quality);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_hdlr(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_hdlr_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_hdlr_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read(fd, &data->type.u32, 4);
    abcdk_mp4_read(fd, &data->subtype.u32, 4);
    abcdk_mp4_read(fd, data->reserved, sizeof(data->reserved));

    nsize = dsize - 1 - 3 - 4 - 4 - 4 * 3;
    if (nsize > 0)
    {
        data->name = abcdk_object_alloc2(nsize + 1);
        if (!data->name)
            goto final_error;

        abcdk_mp4_read(fd, data->name->pptrs[0], nsize);
    }

    return 0;

final_error:

    abcdk_object_unref(&data->name);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_vmhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_vmhd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_vmhd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u16(fd, &data->mode);
    abcdk_mp4_read(fd, &data->opcolor, sizeof(data->opcolor));

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_dref(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_dref_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_dref_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        chk = _abcdk_mp4_read_probe(node ,fd,NULL);
        if(chk == -1)
            goto final_error;
    }

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_stsd(int fd, abcdk_tree_t *node)
{
    abcdk_tree_t *sub_node = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stsd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_stsd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        chk = _abcdk_mp4_read_probe(node ,fd,NULL);
        if(chk == -1)
            goto final_error;
    }

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_stts(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stts_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_stts_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_stts_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_stts_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        for (size_t i = 0; i < data->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, &data->tables[i].sample_count);
            abcdk_mp4_read_u32(fd, &data->tables[i].sample_duration);
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_ctts(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_ctts_t *data = NULL;
    abcdk_mp4_atom_ctts_table_t *table = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_ctts_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_ctts_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_ctts_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        for (size_t i = 0; i < data->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, &data->tables[i].sample_count);
            abcdk_mp4_read_u32(fd, &data->tables[i].composition_offset);
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_stsc(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stsc_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_stsc_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_stsc_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_stsc_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        for (size_t i = 0; i < data->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, &data->tables[i].first_chunk);
            abcdk_mp4_read_u32(fd, &data->tables[i].samples_perchunk);
            abcdk_mp4_read_u32(fd, &data->tables[i].sample_desc_id);
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_stsz(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stsz_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_stsz_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->sample_size);
    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_stsz_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_stsz_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        for (size_t i = 0; i < data->numbers; i++)
        {
            abcdk_mp4_read_u32(fd,&data->tables[i].size);
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_stco(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stco_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_stco_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_stco_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_stco_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STCO)
        {
            for (size_t i = 0; i < data->numbers; i++)
            {
                abcdk_mp4_read_u32to64(fd, &data->tables[i].offset);
            }
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_CO64)
        {
            for (size_t i = 0; i < data->numbers; i++)
            {
                abcdk_mp4_read_u64(fd, &data->tables[i].offset);
            }
        }
        else
        {
            ABCDK_ERRNO_AND_GOTO1(EINVAL,final_error);
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_stss(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stss_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_stss_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_stss_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_stss_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        for (size_t i = 0; i < data->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, &data->tables[i].sync);
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_gmhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);
    
    _abcdk_mp4_read_probe(node,fd,NULL);

    return 0;
}

int _abcdk_mp4_read_smhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_smhd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_smhd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u16(fd, &data->balance);
    abcdk_mp4_read_u16(fd, &data->reserved);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_elst(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_elst_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_elst_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_elst_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_elst_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        for (size_t i = 0; i < data->numbers; i++)
        {
            if(data->version==0)
            {
                abcdk_mp4_read_u32to64(fd, &data->tables[i].track_duration);
                abcdk_mp4_read_u32to64(fd, &data->tables[i].media_time);
            }
            else
            {
                abcdk_mp4_read_u64(fd, &data->tables[i].track_duration);
                abcdk_mp4_read_u64(fd, &data->tables[i].media_time);
            }
            abcdk_mp4_read_u32(fd, &data->tables[i].media_rate);
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_mehd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mehd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_mehd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    if (data->version == 0)
        abcdk_mp4_read_u32to64(fd, &data->duration);
    else
        abcdk_mp4_read_u64(fd, &data->duration);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_trex(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_trex_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_trex_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->trackid);
    abcdk_mp4_read_u32(fd, &data->sample_desc_idx);
    abcdk_mp4_read_u32to64(fd, &data->sample_duration);
    abcdk_mp4_read_u32(fd, &data->sample_size);
    abcdk_mp4_read_u32(fd, &data->sample_flags);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_mfhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mfhd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_mfhd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32to64(fd, &data->sequence_number);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_tfhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_tfhd_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_tfhd_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->trackid);

    if (data->flags & ABCDK_MP4_TFHD_FLAG_BASE_DATA_OFFSET_PRESENT)
    {
        if (data->version == 0)
            abcdk_mp4_read_u32to64(fd, &data->base_data_offset);
        else 
            abcdk_mp4_read_u64(fd, &data->base_data_offset);
    }
    else
        data->base_data_offset = 0;

    if (data->flags & ABCDK_MP4_TFHD_FLAG_SAMPLE_DESCRIPTION_INDEX_PRESENT)
        abcdk_mp4_read_u32(fd, &data->sample_desc_idx);
    else
        data->sample_desc_idx = 1;

    if (data->flags & ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_DURATION_PRESENT)
        abcdk_mp4_read_u32to64(fd, &data->sample_duration);
    else 
        data->sample_duration = 0;

    if (data->flags & ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_SIZE_PRESENT)
        abcdk_mp4_read_u32(fd, &data->sample_size);
    else
        data->sample_size = 0;

    if (data->flags & ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_FLAGS_PRESENT)
        abcdk_mp4_read_u32(fd, &data->sample_flags);
    else 
        data->sample_flags = 0;

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}


int _abcdk_mp4_read_tfdt(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_tfdt_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_tfdt_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    if (data->version == 0)
        abcdk_mp4_read_u32to64(fd, &data->base_decode_time);
    else
        abcdk_mp4_read_u64(fd, &data->base_decode_time);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_trun(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_trun_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    uint32_t discard;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_trun_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->numbers);

    /* 读取已知的选项字段。*/
    if (data->flags & ABCDK_MP4_TRUN_FLAG_DATA_OFFSET_PRESENT)
        abcdk_mp4_read_u32(fd, &data->data_offset);
    if (data->flags & ABCDK_MP4_TRUN_FLAG_FIRST_SAMPLE_FLAGS_PRESENT)
        abcdk_mp4_read_u32(fd, &data->first_sample_flags);

    /* 跳过未知的选项字段。*/
    for (uint32_t i = 0; i < 8; i++)
    {
        uint32_t c = (1 << i);
        uint32_t c2 = data->flags & 0x0000FF;

        /* 跳过未定义的选项字段。*/
        if (!(c2 & c))
            continue;
        
        if(c & ABCDK_MP4_TRUN_FLAG_OPTION_RESERVED)
            abcdk_mp4_read_u32(fd, &discard);
    }

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_trun_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_trun_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        for (size_t i = 0; i < data->numbers; i++)
        {
            /* 读取已知采样表的字段。*/
            if (data->flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_DURATION_PRESENT)
                abcdk_mp4_read_u32(fd, &data->tables[i].sample_duration);
            if (data->flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_SIZE_PRESENT)
                abcdk_mp4_read_u32(fd, &data->tables[i].sample_size);
            if (data->flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_FLAGS_PRESENT)
                abcdk_mp4_read_u32(fd, &data->tables[i].sample_flags);
            if (data->flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_COMPOSITION_TIME_OFFSET_PRESENT)
                abcdk_mp4_read_u32(fd, &data->tables[i].composition_offset);

            /* 跳过未知的采样表的字段。*/
            for (uint32_t i = 0; i < 8; i++)
            {
                uint32_t c = (1 << (i + 8));
                uint32_t c2 = data->flags & 0x00FF00;

                /* 跳过未定义的可选字段。*/
                if (!(c2 & c))
                    continue;
                
                if (c & ABCDK_MP4_TRUN_FLAG_SAPLME_RESERVED)
                    abcdk_mp4_read_u32(fd, &discard);
            }
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}


int _abcdk_mp4_read_mfro(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mfro_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_mfro_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    if (data->version == 0)
        abcdk_mp4_read_u32to64(fd, &data->size);
    else
        abcdk_mp4_read_u64(fd, &data->size);

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_tfra(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_tfra_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_tfra_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    abcdk_mp4_read_u32(fd, &data->trackid);

    uint32_t fields = 0;
    abcdk_mp4_read_u32(fd, &fields);

    data->length_size_traf_num = (fields >> 4) & 3;
    data->length_size_trun_num = (fields >> 2) & 3;
    data->length_size_sample_num = (fields)&3;

    abcdk_mp4_read_u32(fd, &data->numbers);

    if (data->numbers > 0)
    {
        data->tables = (abcdk_mp4_atom_tfra_table_t*)abcdk_heap_alloc(sizeof(abcdk_mp4_atom_tfra_table_t)*data->numbers);
        if (!data->tables)
            goto final_error;

        for (uint32_t i = 0; i < data->numbers; i++)
        {
            if (data->version == 0)
            {
                abcdk_mp4_read_u32to64(fd, &data->tables[i].time);
                abcdk_mp4_read_u32to64(fd, &data->tables[i].moof_offset);
            }
            else
            {
                abcdk_mp4_read_u64(fd, &data->tables[i].time);
                abcdk_mp4_read_u64(fd, &data->tables[i].moof_offset);
            }

            abcdk_mp4_read_nbytes_u32(fd, data->length_size_traf_num, &data->tables[i].traf_number);
            abcdk_mp4_read_nbytes_u32(fd, data->length_size_trun_num, &data->tables[i].trun_number);
            abcdk_mp4_read_nbytes_u32(fd, data->length_size_sample_num, &data->tables[i].sample_number);
        }
    }

    return 0;

final_error:

    abcdk_heap_free(data->tables);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_sample_video(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_sample_desc_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_sample_desc_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read(fd,data->reserved,sizeof(data->reserved));
    abcdk_mp4_read_u16(fd, &data->data_refer_index);

    abcdk_mp4_read_u16(fd, &data->detail.video.reserved1);
    abcdk_mp4_read_u16(fd, &data->detail.video.reserved2);
    abcdk_mp4_read(fd, data->detail.video.reserved3,sizeof(data->detail.video.reserved3));
    abcdk_mp4_read_u16(fd, &data->detail.video.width);
    abcdk_mp4_read_u16(fd, &data->detail.video.height);
    abcdk_mp4_read_u32(fd, &data->detail.video.horiz);
    abcdk_mp4_read_u32(fd, &data->detail.video.vert);
    abcdk_mp4_read_u32(fd, &data->detail.video.reserved4);
    abcdk_mp4_read_u16(fd, &data->detail.video.frame_count);
    abcdk_mp4_read(fd, data->detail.video.encname,32);
    abcdk_mp4_read_u16(fd, &data->detail.video.depth);
    abcdk_mp4_read_u16(fd, &data->detail.video.reserved5);

    /* 如果后面还有跟着的子项，则继续解析。 */
    chk = _abcdk_mp4_read_probe(node, fd, NULL);
    if (chk == -1)
        goto final_error;

    return 0;

final_error:

    memset(data, 0, sizeof(*data));

    return -1;
}


int _abcdk_mp4_read_sample_sound(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_sample_desc_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_sample_desc_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read(fd,data->reserved,sizeof(data->reserved));
    abcdk_mp4_read_u16(fd, &data->data_refer_index);

    abcdk_mp4_read_u16(fd, &data->detail.sound.version);
    abcdk_mp4_read_u16(fd, &data->detail.sound.revision);
    abcdk_mp4_read_u32(fd, &data->detail.sound.reserved1);
    abcdk_mp4_read_u16(fd, &data->detail.sound.channels);
    abcdk_mp4_read_u16(fd, &data->detail.sound.sample_size);
    abcdk_mp4_read_u16(fd, &data->detail.sound.compression_id);
    abcdk_mp4_read_u16(fd, &data->detail.sound.packet_size);
    abcdk_mp4_read_u32(fd, &data->detail.sound.sample_rate);

    if (data->detail.sound.version == 1)
    {
        abcdk_mp4_read_u32(fd, &data->detail.sound.v1.samples_per_packet);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v1.bytes_per_Packet);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v1.bytes_per_frame);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v1.bytes_per_sample);
    }
    else if (data->detail.sound.version == 2)
    {
        abcdk_mp4_read_u32(fd, &data->detail.sound.v2.struct_size);
        abcdk_mp4_read_u64(fd, (uint64_t*)&data->detail.sound.v2.sample_rate);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v2.channels);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v2.reserved);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v2.bits_per_channel);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v2.format_specific_flags);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v2.bytes_per_audio_packet);
        abcdk_mp4_read_u32(fd, &data->detail.sound.v2.lpcm_frames_per_audio_packet);

        if (data->detail.sound.v2.struct_size > 72)
        {
            data->detail.sound.v2.extension = abcdk_object_alloc2(data->detail.sound.v2.struct_size - 72);
            if (!data->detail.sound.v2.extension)
                goto final_error;

            abcdk_mp4_read(fd, data->detail.sound.v2.extension->pptrs[0], data->detail.sound.v2.extension->sizes[0]);
        }
    }

    /* 如果后面还有跟着的子项，则继续解析。 */
    chk = _abcdk_mp4_read_probe(node, fd, NULL);
    if (chk == -1)
        goto final_error;

    return 0;

final_error:

    abcdk_object_unref(&data->detail.sound.v2.extension);
    memset(data, 0, sizeof(*data));

    return -1;
}


int _abcdk_mp4_read_sample_subtitle(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_sample_desc_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_sample_desc_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read(fd,data->reserved,sizeof(data->reserved));
    abcdk_mp4_read_u16(fd, &data->data_refer_index);

    data->detail.subtitle.extension = abcdk_object_alloc2(dsize - 8);
    if (!data->detail.subtitle.extension)
        goto final_error;
    
    abcdk_mp4_read(fd, data->detail.subtitle.extension->pptrs[0], data->detail.subtitle.extension->sizes[0]);

    return 0;

final_error:

    abcdk_object_unref(&data->detail.subtitle.extension);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_avcc(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_avcc_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_avcc_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);
    
    if (dsize <= 0)
        goto final;

    data->extradata = abcdk_object_alloc2(dsize);
    if (!data->extradata)
        goto final_error;

    abcdk_mp4_read(fd, data->extradata->pptrs[0], dsize);

final:

    return 0;

final_error:

    abcdk_object_unref(&data->extradata);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_hvcc(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_hvcc_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_hvcc_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);
    
    if (dsize <= 0)
        goto final;

    data->extradata = abcdk_object_alloc2(dsize);
    if (!data->extradata)
        goto final_error;

    abcdk_mp4_read(fd, data->extradata->pptrs[0], dsize);

final:

    return 0;

final_error:

    abcdk_object_unref(&data->extradata);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_descr_tag(int fd)
{
    uint8_t tag;

    abcdk_mp4_read(fd,&tag,1);

    return tag;
}

int _abcdk_mp4_read_descr_len(int fd)
{
    int c = 0;
    int len = 0;
    int count = 4;

    while (count--) 
    {
        abcdk_mp4_read_u8to32(fd,&c);

        len = (len << 7) | (c & 0x7f);
        if (!(c & 0x80))
            break;
    }

    return len;
}

int _abcdk_mp4_read_esds(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_esds_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int len = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_esds_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&data->version,&data->flags);

    while (1)
    {
        ssize_t cursor = lseek(fd,0,SEEK_CUR);
        if(cursor >= atom->off_head + atom->size)
            break;

        data->tag = _abcdk_mp4_read_descr_tag(fd);
        len = _abcdk_mp4_read_descr_len(fd);

        if (data->tag == ABCDK_MP4_ESDS_ES)
        {
            abcdk_mp4_read_u16(fd,&data->es.id);
            abcdk_mp4_read(fd,&data->es.flags,1);

            if (data->es.flags & 0x80)
                abcdk_mp4_read_u16(fd,&data->es.depends);

            if (data->es.flags & 0x40)
            {
                abcdk_mp4_read(fd,data->es.url,1);
                abcdk_mp4_read(fd, data->es.url + 1, ABCDK_PTR2U8(data->es.url, 0));
            }

            if (data->es.flags & 0x20)
                abcdk_mp4_read_u16(fd,&data->es.ocr);
        }
        else if (data->tag == ABCDK_MP4_ESDS_DEC_CONF)
        {
            abcdk_mp4_read(fd,&data->dec_conf.type_id,1);
            abcdk_mp4_read(fd,&data->dec_conf.stream_type,1);
            abcdk_mp4_read_u24to32(fd,&data->dec_conf.buffer_size);
            abcdk_mp4_read_u32(fd,&data->dec_conf.max_bitrate);
            abcdk_mp4_read_u32(fd,&data->dec_conf.avg_bitrate);
        }
        else if (data->tag == ABCDK_MP4_ESDS_DEC_SP_INFO)
        {
            data->dec_sp_info.extradata = abcdk_object_alloc2(len);
            if (!data->dec_sp_info.extradata)
                goto final_error;

            abcdk_mp4_read(fd,data->dec_sp_info.extradata->pptrs[0], len);
        }
        else if (data->tag == ABCDK_MP4_ESDS_SL_CONF)
        {
            abcdk_mp4_read(fd,&data->dec_ld_conf.reserved,1);
        }
        else
        {
            /*跳过其它的。*/
            lseek(fd,len,SEEK_CUR);
        }
    }

final:

    return 0;

final_error:

    abcdk_object_unref(&data->dec_sp_info.extradata);
    memset(data, 0, sizeof(*data));

    return -1;
}

int _abcdk_mp4_read_ignore(int fd, abcdk_tree_t *node)
{ 
    abcdk_mp4_atom_t *atom = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    return 0;
}

int _abcdk_mp4_read_unknown(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_unknown_t *data = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];
    data = (abcdk_mp4_atom_unknown_t *)&atom->data;

    hsize = atom->off_data - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_data, SEEK_SET);
    
    if (dsize > 0)
    {
#if 0
        data->rawbytes = abcdk_object_alloc2(dsize);
        if (!data->rawbytes)
            goto final_error;

        abcdk_mp4_read(fd, data->rawbytes->pptrs[0], dsize);
#endif
    }

    return 0;

final_error:

    abcdk_object_unref(&data->rawbytes);
    memset(data, 0, sizeof(*data));

    return -1;
}

static struct _abcdk_mp4_read_content_methods
{
    uint32_t type;
    int (*read_content)(int fd, abcdk_tree_t *node);
} abcdk_mp4_read_content_methods[] = {
    {ABCDK_MP4_ATOM_TYPE_FTYP, _abcdk_mp4_read_ftyp},
    {ABCDK_MP4_ATOM_TYPE_MVHD, _abcdk_mp4_read_mvhd},
    {ABCDK_MP4_ATOM_TYPE_UDTA, _abcdk_mp4_read_udta},
    {ABCDK_MP4_ATOM_TYPE_TKHD, _abcdk_mp4_read_tkhd},
    {ABCDK_MP4_ATOM_TYPE_MDHD, _abcdk_mp4_read_mdhd},
    {ABCDK_MP4_ATOM_TYPE_HDLR, _abcdk_mp4_read_hdlr},
    {ABCDK_MP4_ATOM_TYPE_VMHD, _abcdk_mp4_read_vmhd},
    {ABCDK_MP4_ATOM_TYPE_DREF, _abcdk_mp4_read_dref},
    {ABCDK_MP4_ATOM_TYPE_STSD, _abcdk_mp4_read_stsd},
    {ABCDK_MP4_ATOM_TYPE_STTS, _abcdk_mp4_read_stts},
    {ABCDK_MP4_ATOM_TYPE_CTTS, _abcdk_mp4_read_ctts},
    {ABCDK_MP4_ATOM_TYPE_STSC, _abcdk_mp4_read_stsc},
    {ABCDK_MP4_ATOM_TYPE_STSZ, _abcdk_mp4_read_stsz},
    {ABCDK_MP4_ATOM_TYPE_STCO, _abcdk_mp4_read_stco},
    {ABCDK_MP4_ATOM_TYPE_CO64, _abcdk_mp4_read_stco},
    {ABCDK_MP4_ATOM_TYPE_STSS, _abcdk_mp4_read_stss},
    {ABCDK_MP4_ATOM_TYPE_GMHD, _abcdk_mp4_read_gmhd},
    {ABCDK_MP4_ATOM_TYPE_SMHD, _abcdk_mp4_read_smhd},
    {ABCDK_MP4_ATOM_TYPE_ELST, _abcdk_mp4_read_elst},
    {ABCDK_MP4_ATOM_TYPE_MEHD, _abcdk_mp4_read_mehd},
    {ABCDK_MP4_ATOM_TYPE_TREX, _abcdk_mp4_read_trex},
    {ABCDK_MP4_ATOM_TYPE_MFHD, _abcdk_mp4_read_mfhd},
    {ABCDK_MP4_ATOM_TYPE_TFHD, _abcdk_mp4_read_tfhd},
    {ABCDK_MP4_ATOM_TYPE_TFDT, _abcdk_mp4_read_tfdt},
    {ABCDK_MP4_ATOM_TYPE_TRUN, _abcdk_mp4_read_trun},
    {ABCDK_MP4_ATOM_TYPE_MFRO, _abcdk_mp4_read_mfro},
    {ABCDK_MP4_ATOM_TYPE_TFRA, _abcdk_mp4_read_tfra},
    {ABCDK_MP4_ATOM_TYPE_AVC1, _abcdk_mp4_read_sample_video},
    {ABCDK_MP4_ATOM_TYPE_HEV1, _abcdk_mp4_read_sample_video},
    {ABCDK_MP4_ATOM_TYPE_MP4A, _abcdk_mp4_read_sample_sound},
    {ABCDK_MP4_ATOM_TYPE_STPP, _abcdk_mp4_read_sample_subtitle},
    {ABCDK_MP4_ATOM_TYPE_AVCC, _abcdk_mp4_read_avcc},
    {ABCDK_MP4_ATOM_TYPE_HVCC, _abcdk_mp4_read_hvcc},
    {ABCDK_MP4_ATOM_TYPE_ESDS, _abcdk_mp4_read_esds},
    {ABCDK_MP4_ATOM_TYPE_FREE, _abcdk_mp4_read_ignore},
    {ABCDK_MP4_ATOM_TYPE_SKIP, _abcdk_mp4_read_ignore},
    {ABCDK_MP4_ATOM_TYPE_WIDE, _abcdk_mp4_read_ignore},
    {ABCDK_MP4_ATOM_TYPE_MDAT, _abcdk_mp4_read_ignore}
   
};

int abcdk_mp4_read_content(int fd, abcdk_tree_t *node)
{
    int (*_read_content)(int fd, abcdk_tree_t *node) = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    int chk = -1;

    assert(fd >= 0 && node != NULL);

    atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];

    /*可能有数据。*/
    atom->have_data = 1;

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_mp4_read_content_methods); i++)
    {
        if (abcdk_mp4_read_content_methods[i].type != atom->type.u32)
            continue;

        _read_content = abcdk_mp4_read_content_methods[i].read_content;
        break;
    }

    if (_read_content)
        chk = _read_content(fd, node);
    else 
        chk = _abcdk_mp4_read_unknown(fd,node);

    return chk;
}

int _abcdk_mp4_read_probe(abcdk_tree_t *root, int fd, abcdk_mp4_tag_t *stop)
{
    abcdk_mp4_atom_t *root_atom = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_tree_t *node = NULL;
    size_t off_curt = -1UL;
    int keep = 1;
    int chk;

    root_atom = (abcdk_mp4_atom_t *)root->obj->pptrs[0];

    off_curt = lseek(fd, 0, SEEK_CUR);

    /*限制在容器内部解析。*/
    keep = ((off_curt < root_atom->off_head + root_atom->size) ? 1 : 0);

    while (keep)
    {
        node = abcdk_mp4_read_header(fd);
        if (!node)
            goto final_error;

        abcdk_tree_insert2(root, node, 0);

        atom = (abcdk_mp4_atom_t *)node->obj->pptrs[0];

        /*统一标记为无数据(容器)。*/
        atom->have_data = 0;
        
        switch (atom->type.u32)
        {
        case ABCDK_MP4_ATOM_TYPE_MOOV:
        case ABCDK_MP4_ATOM_TYPE_TRAK:
        case ABCDK_MP4_ATOM_TYPE_EDTS:
        case ABCDK_MP4_ATOM_TYPE_MDIA:
        case ABCDK_MP4_ATOM_TYPE_MINF:
        case ABCDK_MP4_ATOM_TYPE_DINF:
        case ABCDK_MP4_ATOM_TYPE_STBL:
        case ABCDK_MP4_ATOM_TYPE_MVEX:
        case ABCDK_MP4_ATOM_TYPE_MOOF:
        case ABCDK_MP4_ATOM_TYPE_TRAF:
        case ABCDK_MP4_ATOM_TYPE_MFRA:
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_data, SEEK_SET);

            chk = _abcdk_mp4_read_probe(node, fd, stop);
            if (chk != 0)
                goto final_error;
        }
        break;

        default:
        {
            chk = abcdk_mp4_read_content(fd, node);
            if (chk == -1)
                goto final_error;
        }
            break;
        }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*遇到中断tag，则提前终止。*/
        if(stop && stop->u32 == atom->type.u32)
            break;

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size < root_atom->off_head + root_atom->size) ? 1 : 0);
    }

    return 0;

final_error:

    return -1;
}

abcdk_tree_t *abcdk_mp4_read_probe(int fd, uint64_t offset, uint64_t size, abcdk_mp4_tag_t *stop)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_tree_t *root;
    uint64_t fsize = 0;
    uint64_t offcur = 0;
    int chk;

    assert(fd >= 0 && offset < -1UL && size >= 8);

    chk = abcdk_mp4_size(fd, &fsize);
    if (chk != 0)
        return NULL;

    /*最小的atom为8字节，文件不能比8还小。*/
    if (fsize < 8)
        ABCDK_ERRNO_AND_RETURN1(ESPIPE, NULL);

    /*偏移量不能超过文件末尾。*/
    if (offset >= fsize)
        ABCDK_ERRNO_AND_RETURN1(ESPIPE, NULL);

    /*偏移量到文件末尾之间的数据必须大于8字节。*/
    if (fsize - offset < 8)
        ABCDK_ERRNO_AND_RETURN1(ESPIPE, NULL);

    /*修正最大值，不能超过文件末尾。*/
    if (size > fsize - offset)
        size = fsize - offset;

    /*移动文件指针到指定位置。*/
    if (lseek(fd, offset, SEEK_SET) == -1UL)
        return NULL;

    root = abcdk_tree_alloc3(sizeof(abcdk_mp4_atom_t));
    if (!root)
        return NULL;

    atom = (abcdk_mp4_atom_t *)root->obj->pptrs[0];

    atom->size = size;
    atom->type.u32 = ABCDK_MP4_ATOM_MKTAG('.', '.', '.', '.');
    atom->off_head = atom->off_data = offset;

    _abcdk_mp4_read_probe(root, fd, stop);

    return root;
}

abcdk_tree_t *abcdk_mp4_read_probe2(int fd, uint64_t offset, uint64_t size, uint32_t stop)
{
    abcdk_mp4_tag_t tag;

    assert(fd >= 0 && offset < -1UL && size >= 8);

    tag.u32 = stop;

    return abcdk_mp4_read_probe(fd, offset, size, stop ? &tag : NULL);
}
