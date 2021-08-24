/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-mp4/demuxer.h"

/*
 * 读取atom头，但不改变文件指针位置。
*/
abcdk_tree_t *abcdk_mp4_read_header(int fd)
{
    uint32_t size32 = 0;
    uint64_t size64 = 0;
    uint64_t fsize = 0;
    abcdk_tree_t *node;
    abcdk_mp4_atom_t *atom = NULL;
    int chk;

    node = abcdk_mp4_alloc();
    if (!node)
        return NULL;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];

    /* 保存数据偏移量。*/
    atom->off_head = lseek(fd, 0, SEEK_CUR);

    chk = abcdk_mp4_read_u32(fd, &size32);
    if (chk != 0)
        goto final_error;

    chk = abcdk_mp4_read(fd, &atom->type.u32, sizeof(atom->type.u32));
    if (chk != 0)
        goto final_error;

    /* 当size32==1时，需要读取扩展字段，确定长度。*/
    if (size32 == 1)
    {
        chk = abcdk_mp4_read_u64(fd, &size64);
        if (chk != 0)
            goto final_error;

        atom->size = size64;
        atom->off_cont = atom->off_head + 16;
    }
    else if (size32 == 0)
    {
        chk = abcdk_mp4_size(fd, &fsize);
        if (chk != 0)
            goto final_error;

        atom->size = fsize - atom->off_head;
        atom->off_cont = atom->off_head + 8;
    }
    else
    {
        atom->size = size32;
        atom->off_cont = atom->off_head + 8;
    }

    /* 恢复偏移量。*/
    lseek(fd, atom->off_head, SEEK_SET);

    return node;

final_error:

    abcdk_tree_free(&node);

    return NULL;
}

int abcdk_mp4_read_fullheader(int fd, uint8_t *ver, uint32_t *flags)
{
    uint32_t h = 0;
    if (abcdk_mp4_read_u32(fd, &h))
        return -1;

    *ver = ((h >> 24) & 0x000000FF);
    *flags = (h & 0x00FFFFFF);

    return 0;
}

int _abcdk_mp4_read_probe(abcdk_tree_t *root, int fd, abcdk_mp4_tag_t *stop)
{
    abcdk_mp4_atom_t *root_atom = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_tree_t *node = NULL;
    int keep = 1;
    int chk;

    root_atom = (abcdk_mp4_atom_t *)root->alloc->pptrs[0];

    while (keep)
    {
        node = abcdk_mp4_read_header(fd);
        if (!node)
            goto final_error;

        abcdk_tree_insert2(root, node, 0);

        atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
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
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe(node, fd, stop);
            if (chk != 0)
                goto final_error;
        }
        break;

        default:
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

    atom = (abcdk_mp4_atom_t *)root->alloc->pptrs[0];

    atom->size = size;
    atom->type.u32 = ABCDK_MP4_ATOM_MKTAG('!', '@', '#', '$');
    atom->off_head = atom->off_cont = offset;

    _abcdk_mp4_read_probe(root, fd, stop);

    return root;
}

abcdk_tree_t *abcdk_mp4_read_probe2(int fd, uint64_t offset, uint64_t size, uint32_t stop)
{
    abcdk_mp4_tag_t tag = {0};

    assert(fd >= 0 && offset < -1UL && size >= 8);

    tag.u32 = stop;

    return abcdk_mp4_read_probe(fd, offset, size, stop ? &tag : NULL);
}

int _abcdk_mp4_read_ftyp(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_ftyp_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_ftyp_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_ftyp_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->major.u32, 4);
    abcdk_mp4_read_u32(fd, &cont->minor);

    /*兼容数据不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->compat);

    cont->compat = abcdk_allocator_alloc3(4, (dsize - 8) / 4);
    if (!cont->compat)
        goto final_error;

    for (size_t i = 0; i < cont->compat->numbers; i++)
        abcdk_mp4_read(fd, cont->compat->pptrs[i], 4);

    return 0;

final_error:

    abcdk_allocator_unref(&cont->compat);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_mvhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mvhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_mvhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_mvhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    if (cont->version == 0)
    {
        abcdk_mp4_read_u32to64(fd, &cont->ctime);
        abcdk_mp4_read_u32to64(fd, &cont->mtime);
        abcdk_mp4_read_u32(fd, &cont->timescale);
        abcdk_mp4_read_u32to64(fd, &cont->duration);
    }
    else
    {
        abcdk_mp4_read_u64(fd, &cont->ctime);
        abcdk_mp4_read_u64(fd, &cont->mtime);
        abcdk_mp4_read_u32(fd, &cont->timescale);
        abcdk_mp4_read_u64(fd, &cont->duration);
    }

    abcdk_mp4_read_u32(fd, &cont->rate);
    abcdk_mp4_read_u16(fd, &cont->volume);
    abcdk_mp4_read(fd, cont->reserved, 10);

    for (int i = 0; i < 9; i++)
        abcdk_mp4_read_u32(fd, &cont->matrix[i]);

    abcdk_mp4_read(fd, cont->predefined, sizeof(cont->predefined));
    abcdk_mp4_read_u32(fd, &cont->nexttrackid);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_udta(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    
    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd,atom->off_cont,SEEK_SET);

    atom->entries = abcdk_mp4_read_probe(fd, atom->off_cont, dsize,NULL);
    if(!atom->entries)
        return -1;
    
    return 0;
}

int _abcdk_mp4_read_tkhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_tkhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_tkhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_tkhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    if (cont->version == 0)
    {
        abcdk_mp4_read_u32to64(fd, &cont->ctime);
        abcdk_mp4_read_u32to64(fd, &cont->mtime);
        abcdk_mp4_read_u32(fd, &cont->trackid);
        abcdk_mp4_read_u32(fd, &cont->reserved1);
        abcdk_mp4_read_u32to64(fd, &cont->duration);
    }
    else
    {
        abcdk_mp4_read_u64(fd, &cont->ctime);
        abcdk_mp4_read_u64(fd, &cont->mtime);
        abcdk_mp4_read_u32(fd, &cont->trackid);
        abcdk_mp4_read_u32(fd, &cont->reserved1);
        abcdk_mp4_read_u64(fd, &cont->duration);
    }

    abcdk_mp4_read_u64(fd, &cont->reserved2);
    abcdk_mp4_read_u16(fd, &cont->layer);
    abcdk_mp4_read_u16(fd, &cont->alternategroup);
    abcdk_mp4_read_u16(fd, &cont->volume);
    abcdk_mp4_read_u16(fd, &cont->reserved3);

    for (int i = 0; i < 9; i++)
        abcdk_mp4_read_u32(fd, &cont->matrix[i]);

    abcdk_mp4_read_u32(fd, &cont->width);
    abcdk_mp4_read_u32(fd, &cont->height);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_mdhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mdhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_mdhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_mdhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    if (cont->version == 0)
    {
        abcdk_mp4_read_u32to64(fd, &cont->ctime);
        abcdk_mp4_read_u32to64(fd, &cont->mtime);
        abcdk_mp4_read_u32(fd, &cont->timescale);
        abcdk_mp4_read_u32to64(fd, &cont->duration);
    }
    else
    {
        abcdk_mp4_read_u64(fd, &cont->ctime);
        abcdk_mp4_read_u64(fd, &cont->mtime);
        abcdk_mp4_read_u32(fd, &cont->timescale);
        abcdk_mp4_read_u64(fd, &cont->duration);
    }

    abcdk_mp4_read_u16(fd, &cont->language);
    abcdk_mp4_read_u16(fd, &cont->quality);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_hdlr(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_hdlr_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_hdlr_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_hdlr_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read(fd, &cont->type.u32, 4);
    abcdk_mp4_read(fd, &cont->subtype.u32, 4);
    abcdk_mp4_read(fd, cont->reserved, sizeof(cont->reserved));

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->name);

    nsize = dsize - 1 - 3 - 4 - 4 - 4 * 3;
    if (nsize > 0)
    {
        cont->name = abcdk_allocator_alloc2(nsize + 1);
        if (!cont->name)
            goto final_error;

        abcdk_mp4_read(fd, cont->name->pptrs[0], nsize);
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->name);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_vmhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_vmhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_vmhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_vmhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u16(fd, &cont->mode);
    abcdk_mp4_read(fd, &cont->opcolor, sizeof(cont->opcolor));

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_dref(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_dref_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_dref_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_dref_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    if (cont->numbers > 0)
    {
        atom->entries = abcdk_mp4_read_probe(fd, atom->off_cont + 8, dsize - 8, NULL);
        if(!atom->entries)
            goto final_error;
    }

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_stsd(int fd, abcdk_tree_t *node)
{
    abcdk_tree_t *sub_node = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stsd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stsd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stsd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    if (cont->numbers > 0)
    {
        atom->entries = abcdk_mp4_read_probe(fd, atom->off_cont + 8, dsize - 8, NULL);
        if (!atom->entries)
            goto final_error;
    }

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_stts(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stts_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stts_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stts_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        cont->tables = abcdk_allocator_alloc3(8, cont->numbers);
        if (!cont->tables)
            goto final_error;

        for (size_t i = 0; i < cont->tables->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 0));
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 4));
        }
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_ctts(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_ctts_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_ctts_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_ctts_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        cont->tables = abcdk_allocator_alloc3(8, cont->numbers);
        if (!cont->tables)
            goto final_error;

        for (size_t i = 0; i < cont->tables->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 0));
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 4));
        }
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_stsc(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stsc_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stsc_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stsc_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        cont->tables = abcdk_allocator_alloc3(12, cont->numbers);
        if (!cont->tables)
            goto final_error;

        for (size_t i = 0; i < cont->tables->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 0));
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 4));
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 8));
        }
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_stsz(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stsz_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stsz_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stsz_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->samplesize);
    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        cont->tables = abcdk_allocator_alloc3(4, cont->numbers);
        if (!cont->tables)
            goto final_error;

        for (size_t i = 0; i < cont->tables->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 0));
        }
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_stco(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stco_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stco_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stco_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STCO)
        {
            if (cont->numbers > dsize / 4)
                cont->numbers = dsize / 4;

            cont->tables = abcdk_allocator_alloc3(4, cont->numbers);
            if (!cont->tables)
                goto final_error;

            for (size_t i = 0; i < cont->tables->numbers; i++)
            {
                abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 0));
            }
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_CO64)
        {
            if (cont->numbers > dsize / 8)
                cont->numbers = dsize / 8;

            cont->tables = abcdk_allocator_alloc3(8, cont->numbers);
            if (!cont->tables)
                goto final_error;

            for (size_t i = 0; i < cont->tables->numbers; i++)
            {
                abcdk_mp4_read_u64(fd, ABCDK_PTR2U64PTR(cont->tables->pptrs[i], 0));
            }
        }
        else
        {
            ABCDK_ERRNO_AND_GOTO1(EINVAL,final_error);
        }
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_stss(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_stss_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stss_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stss_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        cont->tables = abcdk_allocator_alloc3(4, cont->numbers);
        if (!cont->tables)
            goto final_error;

        for (size_t i = 0; i < cont->tables->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 0));
        }
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_gmhd(int fd, abcdk_tree_t *node)
{
    abcdk_tree_t *sub_root = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    sub_root = abcdk_mp4_read_probe(fd, atom->off_cont, dsize, NULL);
    if(!sub_root)
        return -1;

    abcdk_tree_insert2(node,sub_root,0);

    return 0;
}

int _abcdk_mp4_read_smhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_smhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_smhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_smhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u16(fd, &cont->balance);
    abcdk_mp4_read_u16(fd, &cont->reserved);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_elst(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_elst_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_elst_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_elst_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        cont->tables = abcdk_allocator_alloc3(12, cont->numbers);
        if (!cont->tables)
            goto final_error;

        for (size_t i = 0; i < cont->tables->numbers; i++)
        {
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 0));
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 4));
            abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 8));
        }
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_mehd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mehd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_mehd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_mehd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    if (cont->version == 0)
        abcdk_mp4_read_u32to64(fd, &cont->duration);
    else
        abcdk_mp4_read_u64(fd, &cont->duration);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_trex(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_trex_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_trex_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_trex_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->trackid);
    abcdk_mp4_read_u32(fd, &cont->default_sample_desc_index);
    abcdk_mp4_read_u32to64(fd, &cont->default_duration);
    abcdk_mp4_read_u32(fd, &cont->default_samplesize);
    abcdk_mp4_read_u32(fd, &cont->default_sampleflags);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_mfhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mfhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_mfhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_mfhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32to64(fd, &cont->sn);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_tfhd(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_tfhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_tfhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_tfhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->trackid);

    if (cont->flags & ABCDK_MP4_TFHD_FLAG_BASE_DATA_OFFSET_PRESENT)
    {
        if (cont->version == 0)
            abcdk_mp4_read_u32to64(fd, &cont->base_data_offset);
        else 
            abcdk_mp4_read_u64(fd, &cont->base_data_offset);
    }
    else
        cont->base_data_offset = 0;

    if (cont->flags & ABCDK_MP4_TFHD_FLAG_SAMPLE_DESCRIPTION_INDEX_PRESENT)
        abcdk_mp4_read_u32(fd, &cont->sample_desc_index);
    else
        cont->sample_desc_index = 1;

    if (cont->flags & ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_DURATION_PRESENT)
        abcdk_mp4_read_u32to64(fd, &cont->default_duration);
    else 
        cont->default_duration = 0;

    if (cont->flags & ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_SIZE_PRESENT)
        abcdk_mp4_read_u32(fd, &cont->default_samplesize);
    else
        cont->default_samplesize = 0;

    if (cont->flags & ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_FLAGS_PRESENT)
        abcdk_mp4_read_u32(fd, &cont->default_sampleflags);
    else 
        cont->default_sampleflags = 0;

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}


int _abcdk_mp4_read_tfdt(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_tfdt_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_tfdt_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_tfdt_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    if (cont->version == 0)
        abcdk_mp4_read_u32to64(fd, &cont->base_decode_time);
    else
        abcdk_mp4_read_u64(fd, &cont->base_decode_time);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_trun(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_trun_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    uint32_t discard;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_trun_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_trun_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /* 读取已知的选项字段。*/
    if (cont->flags & ABCDK_MP4_TRUN_FLAG_DATA_OFFSET_PRESENT)
        abcdk_mp4_read_u32(fd, &cont->data_offset);
    if (cont->flags & ABCDK_MP4_TRUN_FLAG_FIRST_SAMPLE_FLAGS_PRESENT)
        abcdk_mp4_read_u32(fd, &cont->first_sample_flags);

    /* 跳过未知的选项字段。*/
    for (uint32_t i = 0; i < 8; i++)
    {
        uint32_t c = (1 << i);
        uint32_t c2 = cont->flags & 0x0000FF;

        /* 跳过未定义的选项字段。*/
        if (!(c2 & c))
            continue;
        
        if(c & ABCDK_MP4_TRUN_FLAG_OPTION_RESERVED)
            abcdk_mp4_read_u32(fd, &discard);
    }

    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        cont->tables = abcdk_allocator_alloc3(16, cont->numbers);
        if (!cont->tables)
            goto final_error;

        for (size_t i = 0; i < cont->tables->numbers; i++)
        {
            /* 读取已知采样表的字段。*/
            if (cont->flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_DURATION_PRESENT)
                abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 0));
            if (cont->flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_SIZE_PRESENT)
                abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 4));
            if (cont->flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_FLAGS_PRESENT)
                abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 8));
            if (cont->flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_COMPOSITION_TIME_OFFSET_PRESENT)
                abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 12));

            /* 跳过未知的采样表的字段。*/
            for (uint32_t i = 0; i < 8; i++)
            {
                uint32_t c = (1 << (i + 8));
                uint32_t c2 = cont->flags & 0x00FF00;

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

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}


int _abcdk_mp4_read_mfro(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_mfro_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_mfro_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_mfro_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    if (cont->version == 0)
        abcdk_mp4_read_u32to64(fd, &cont->size);
    else
        abcdk_mp4_read_u64(fd, &cont->size);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_tfra(int fd, abcdk_tree_t *node)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_tfra_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_tfra_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_tfra_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read_fullheader(fd,&cont->version,&cont->flags);

    abcdk_mp4_read_u32(fd, &cont->trackid);

    uint32_t fields = 0;
    abcdk_mp4_read_u32(fd, &fields);

    cont->length_size_traf_num = (fields >> 4) & 3;
    cont->length_size_trun_num = (fields >> 2) & 3;
    cont->length_size_sample_num = (fields)&3;

    abcdk_mp4_read_u32(fd, &cont->numbers);


    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->tables);

    if (cont->numbers > 0)
    {
        cont->tables = abcdk_allocator_alloc3(28, cont->numbers);
        if (!cont->tables)
            goto final_error;

        for (uint32_t i = 0; i < cont->numbers; i++)
        {
            if (cont->version == 0)
            {
                abcdk_mp4_read_u32to64(fd, ABCDK_PTR2U64PTR(cont->tables->pptrs[i], 0));
                abcdk_mp4_read_u32to64(fd, ABCDK_PTR2U64PTR(cont->tables->pptrs[i], 8));
            }
            else
            {
                abcdk_mp4_read_u64(fd, ABCDK_PTR2U64PTR(cont->tables->pptrs[i], 0));
                abcdk_mp4_read_u64(fd, ABCDK_PTR2U64PTR(cont->tables->pptrs[i], 8));
            }

            switch (cont->length_size_traf_num)
            {
            case 0:
                abcdk_mp4_read_u8to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 16));
                break;
            case 1:
                abcdk_mp4_read_u16to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 16));
                break;
            case 2:
                abcdk_mp4_read_u24to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 16));
                break;
            case 3:
                abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 16));
                break;
            }

            switch (cont->length_size_trun_num)
            {
            case 0:
                abcdk_mp4_read_u8to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 20));
                break;
            case 1:
                abcdk_mp4_read_u16to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 20));
                break;
            case 2:
                abcdk_mp4_read_u24to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 20));
                break;
            case 3:
                abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 20));
                break;
            }

            switch (cont->length_size_sample_num)
            {
            case 0:
                abcdk_mp4_read_u8to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 24));
                break;
            case 1:
                abcdk_mp4_read_u16to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 24));
                break;
            case 2:
                abcdk_mp4_read_u24to32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 24));
                break;
            case 3:
                abcdk_mp4_read_u32(fd, ABCDK_PTR2U32PTR(cont->tables->pptrs[i], 24));
                break;
            }
        }
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->tables);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_sample_video(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_tree_t *sub_node = NULL;
    abcdk_mp4_atom_t *sub_atom = NULL;
    abcdk_mp4_atom_sample_desc_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_sample_desc_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_sample_desc_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd,cont->reserved,sizeof(cont->reserved));
    abcdk_mp4_read_u16(fd, &cont->data_refer_index);

    abcdk_mp4_read_u16(fd, &cont->detail.video.reserved1);
    abcdk_mp4_read_u16(fd, &cont->detail.video.reserved2);
    abcdk_mp4_read(fd, cont->detail.video.reserved3,sizeof(cont->detail.video.reserved3));
    abcdk_mp4_read_u16(fd, &cont->detail.video.width);
    abcdk_mp4_read_u16(fd, &cont->detail.video.height);
    abcdk_mp4_read_u32(fd, &cont->detail.video.horiz);
    abcdk_mp4_read_u32(fd, &cont->detail.video.vert);
    abcdk_mp4_read_u32(fd, &cont->detail.video.reserved4);
    abcdk_mp4_read_u16(fd, &cont->detail.video.frame_count);
    abcdk_mp4_read(fd, cont->detail.video.encname,32);
    abcdk_mp4_read_u16(fd, &cont->detail.video.depth);
    abcdk_mp4_read_u16(fd, &cont->detail.video.reserved5);

    /* 如果后面还有跟着的子项，则继续解析。 */
    atom->entries = abcdk_mp4_read_probe(fd, atom->off_cont + 78, dsize - 78, NULL);

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}


int _abcdk_mp4_read_sample_sound(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_tree_t *sub_node = NULL;
    abcdk_mp4_atom_sample_desc_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_sample_desc_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_sample_desc_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd,cont->reserved,sizeof(cont->reserved));
    abcdk_mp4_read_u16(fd, &cont->data_refer_index);

    abcdk_mp4_read_u16(fd, &cont->detail.sound.version);
    abcdk_mp4_read_u16(fd, &cont->detail.sound.revision);
    abcdk_mp4_read_u32(fd, &cont->detail.sound.reserved1);
    abcdk_mp4_read_u16(fd, &cont->detail.sound.channels);
    abcdk_mp4_read_u16(fd, &cont->detail.sound.sample_size);
    abcdk_mp4_read_u16(fd, &cont->detail.sound.compression_id);
    abcdk_mp4_read_u16(fd, &cont->detail.sound.packet_size);
    abcdk_mp4_read_u32(fd, &cont->detail.sound.sample_rate);

    if (cont->detail.sound.version == 1)
    {
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v1.samples_per_packet);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v1.bytes_per_Packet);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v1.bytes_per_frame);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v1.bytes_per_sample);

        /* 如果后面还有跟着的子项，则继续解析。 */
        atom->entries = abcdk_mp4_read_probe(fd, atom->off_cont + 40, dsize - 40, NULL);
    }
    else if (cont->detail.sound.version == 2)
    {
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v2.struct_size);
        abcdk_mp4_read_u64(fd, (uint64_t*)&cont->detail.sound.v2.sample_rate);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v2.channels);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v2.reserved);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v2.bits_per_channel);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v2.format_specific_flags);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v2.bytes_per_audio_packet);
        abcdk_mp4_read_u32(fd, &cont->detail.sound.v2.lpcm_frames_per_audio_packet);

        if (cont->detail.sound.v2.struct_size > 72)
        {
            /*不能利旧，需要册除后重新创建。*/
            abcdk_allocator_unref(&cont->detail.sound.v2.extension);

            cont->detail.sound.v2.extension = abcdk_allocator_alloc2(cont->detail.sound.v2.struct_size - 72);
            if (!cont->detail.sound.v2.extension)
                goto final_error;

            abcdk_mp4_read(fd, cont->detail.sound.v2.extension->pptrs[0], cont->detail.sound.v2.extension->sizes[0]);
        }

        /* 如果后面还有跟着的子项，则继续解析。 */
        atom->entries = abcdk_mp4_read_probe(fd, atom->off_cont + cont->detail.sound.v2.struct_size, dsize - cont->detail.sound.v2.struct_size, NULL);
    }
    else 
    {
        /* 如果后面还有跟着的子项，则继续解析。 */
        atom->entries = abcdk_mp4_read_probe(fd, atom->off_cont + 28, dsize - 28, NULL);
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->detail.sound.v2.extension);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_read_glbl(int fd, abcdk_tree_t *node)
{   
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_mp4_atom_glbl_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
   
    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_glbl_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_glbl_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);
    
    /*不能利旧，需要册除后重新创建。*/
    abcdk_allocator_unref(&cont->extradata);

    if (dsize > 0)
    {
        cont->extradata = abcdk_allocator_alloc2(dsize);
        if (!cont->extradata)
            goto final_error;

        abcdk_mp4_read(fd, cont->extradata->pptrs[0], dsize);
    }

    return 0;

final_error:

    abcdk_allocator_unref(&cont->extradata);
    memset(cont, 0, sizeof(*cont));

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
    {ABCDK_MP4_ATOM_TYPE_GLBL, _abcdk_mp4_read_glbl},
    {ABCDK_MP4_ATOM_TYPE_AVCC, _abcdk_mp4_read_glbl},
    {ABCDK_MP4_ATOM_TYPE_HVCC, _abcdk_mp4_read_glbl},
    {ABCDK_MP4_ATOM_TYPE_DVH1, _abcdk_mp4_read_glbl},
    {ABCDK_MP4_ATOM_TYPE_PRIV, _abcdk_mp4_read_glbl},
    {ABCDK_MP4_ATOM_TYPE_ALIS, _abcdk_mp4_read_glbl}
   
};

int abcdk_mp4_read_content(int fd, abcdk_tree_t *node)
{
    int (*read_content)(int fd, abcdk_tree_t *node) = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    int chk = -1;

    assert(fd >= 0 && node != NULL);

    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_mp4_read_content_methods); i++)
    {
        if (abcdk_mp4_read_content_methods[i].type != atom->type.u32)
            continue;

        read_content = abcdk_mp4_read_content_methods[i].read_content;
        break;
    }

    if (read_content)
        chk = read_content(fd, node);
    else 
        chk = -2;
    
    /*递归子项。*/
    if(atom->entries)
        abcdk_mp4_read_content2(fd,atom->entries);

    return chk;
}

typedef struct _abcdk_mp4_read_content2_ctx
{
    int fd;
    int chk;
} abcdk_mp4_read_content2_ctx_t;

int _abcdk_mp4_read_content2_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_mp4_read_content2_ctx_t *ctx;
    abcdk_mp4_atom_t *atom = NULL;

    if (depth == -1)
        return -1;

    ctx = (abcdk_mp4_read_content2_ctx_t *)opaque;
    atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];

    /*自定义的类型，忽略。*/
    if (atom->type.u32 == ABCDK_MP4_ATOM_MKTAG('!', '@', '#', '$'))
        return 1;

    ctx->chk = abcdk_mp4_read_content(ctx->fd, node);
    if (ctx->chk == -1)
        return -1;

    return 1;
}

int abcdk_mp4_read_content2(int fd, abcdk_tree_t *root)
{
    abcdk_mp4_read_content2_ctx_t ctx;

    assert(fd >= 0 && root != NULL);

    ctx.fd = fd;
    ctx.chk = -1;

    abcdk_tree_iterator_t it = {0, _abcdk_mp4_read_content2_cb, &ctx};
    abcdk_tree_scan(root, &it);

    return ctx.chk;
}