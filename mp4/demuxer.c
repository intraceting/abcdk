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
abcdk_tree_t *_abcdk_mp4_read_atom_header(int fd)
{
    uint32_t size32 = 0;
    uint64_t size64 = 0;
    uint64_t fsize = 0;
    abcdk_tree_t *node;
    abcdk_mp4_atom_t *atom = NULL;
    int chk;

    node = abcdk_tree_alloc3(sizeof(abcdk_mp4_atom_t));
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

int _abcdk_mp4_read_probe(abcdk_tree_t *root, int fd)
{
    abcdk_mp4_atom_t *root_atom = NULL;
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_tree_t *node = NULL;
    int keep = 1;
    int chk;

    root_atom = (abcdk_mp4_atom_t *)root->alloc->pptrs[0];

    while (keep)
    {
        node = _abcdk_mp4_read_atom_header(fd);
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

            chk = _abcdk_mp4_read_probe(node, fd);
            if (chk != 0)
                goto final_error;
        }
        break;

        default:
            break;
        }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size < root_atom->off_head + root_atom->size) ? 1 : 0);
    }

    return 0;

final_error:

    return -1;
}

abcdk_tree_t *abcdk_mp4_read_probe(int fd, uint64_t offset, uint64_t size)
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

    _abcdk_mp4_read_probe(root, fd);

    return root;
}

int _abcdk_mp4_atom_read_ftyp(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_ftyp_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

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

int _abcdk_mp4_atom_read_mvhd(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_mvhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_mvhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_mvhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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


int _abcdk_mp4_atom_read_udta(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_udta_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_udta_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_udta_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    cont->entries = abcdk_mp4_read_probe(fd, atom->off_cont, dsize);

    return 0;

final_error:

    abcdk_tree_free(&cont->entries);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_atom_read_tkhd(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_tkhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_tkhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_tkhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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

int _abcdk_mp4_atom_read_mdhd(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_mdhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_mdhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_mdhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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

int _abcdk_mp4_atom_read_hdlr(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_hdlr_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_hdlr_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_hdlr_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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

int _abcdk_mp4_atom_read_vmhd(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_vmhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_vmhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_vmhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

    abcdk_mp4_read_u16(fd, &cont->mode);
    abcdk_mp4_read(fd, &cont->opcolor, sizeof(cont->opcolor));

    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_atom_read_dref(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_dref_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_dref_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_dref_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_tree_free(&cont->entries);

    if (cont->numbers > 0)
        cont->entries = abcdk_mp4_read_probe(fd, atom->off_cont + 8, dsize - 8);

    return 0;

final_error:

    abcdk_tree_free(&cont->entries);
    memset(cont, 0, sizeof(*cont));

    return -1;
}


int _abcdk_mp4_atom_read_stsd(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_stsd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stsd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stsd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

    abcdk_mp4_read_u32(fd, &cont->numbers);

    /*不能利旧，需要册除后重新创建。*/
    abcdk_tree_free(&cont->entries);

    if (cont->numbers > 0)
        cont->entries = abcdk_mp4_read_probe(fd, atom->off_cont + 8, dsize - 8);

    return 0;

final_error:

    abcdk_tree_free(&cont->entries);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_atom_read_stts(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_stts_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stts_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stts_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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

int _abcdk_mp4_atom_read_ctts(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_ctts_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_ctts_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_ctts_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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

int _abcdk_mp4_atom_read_stsc(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_stsc_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stsc_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stsc_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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

int _abcdk_mp4_atom_read_stsz(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_stsz_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stsz_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stsz_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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


int _abcdk_mp4_atom_read_stco(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_stco_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stco_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stco_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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


int _abcdk_mp4_atom_read_stss(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_stss_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_stss_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_stss_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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

int _abcdk_mp4_atom_read_gmhd(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_gmhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_gmhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_gmhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    cont->entries = abcdk_mp4_read_probe(fd, atom->off_cont, dsize);

    return 0;

final_error:

    abcdk_tree_free(&cont->entries);
    memset(cont, 0, sizeof(*cont));

    return -1;
}

int _abcdk_mp4_atom_read_smhd(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_smhd_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_smhd_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_smhd_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

    abcdk_mp4_read_u16(fd, &cont->balance);
    abcdk_mp4_read_u16(fd, &cont->reserved);
    
    return 0;

final_error:

    memset(cont, 0, sizeof(*cont));

    return -1;
}


int _abcdk_mp4_atom_read_elst(int fd, abcdk_mp4_atom_t *atom)
{
    abcdk_mp4_atom_elst_t *cont = NULL;
    size_t hsize = 0, dsize = 0;
    size_t nsize = 0;
    int chk;

    if (!atom->cont)
        atom->cont = abcdk_allocator_alloc2(sizeof(abcdk_mp4_atom_elst_t));

    if (!atom->cont)
        goto final_error;

    cont = (abcdk_mp4_atom_elst_t *)atom->cont->pptrs[0];

    hsize = atom->off_cont - atom->off_head;
    dsize = atom->size - hsize;

    lseek(fd, atom->off_cont, SEEK_SET);

    abcdk_mp4_read(fd, &cont->version, 1);
    abcdk_mp4_read_u24to32(fd, &cont->flags);

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

static struct _abcdk_mp4_atom_read_content_methods
{
    uint32_t type;
    int (*read_content)(int fd, abcdk_mp4_atom_t *atom);
} abcdk_mp4_atom_read_content_methods[] = {
    {ABCDK_MP4_ATOM_TYPE_FTYP, _abcdk_mp4_atom_read_ftyp},
    {ABCDK_MP4_ATOM_TYPE_MVHD, _abcdk_mp4_atom_read_mvhd},
    {ABCDK_MP4_ATOM_TYPE_UDTA, _abcdk_mp4_atom_read_udta},
    {ABCDK_MP4_ATOM_TYPE_TKHD, _abcdk_mp4_atom_read_tkhd},
    {ABCDK_MP4_ATOM_TYPE_MDHD, _abcdk_mp4_atom_read_mdhd},
    {ABCDK_MP4_ATOM_TYPE_HDLR, _abcdk_mp4_atom_read_hdlr},
    {ABCDK_MP4_ATOM_TYPE_VMHD, _abcdk_mp4_atom_read_vmhd},
    {ABCDK_MP4_ATOM_TYPE_DREF, _abcdk_mp4_atom_read_dref},
    {ABCDK_MP4_ATOM_TYPE_STSD, _abcdk_mp4_atom_read_stsd},
    {ABCDK_MP4_ATOM_TYPE_STTS, _abcdk_mp4_atom_read_stts},
    {ABCDK_MP4_ATOM_TYPE_CTTS, _abcdk_mp4_atom_read_ctts},
    {ABCDK_MP4_ATOM_TYPE_STSC, _abcdk_mp4_atom_read_stsc},
    {ABCDK_MP4_ATOM_TYPE_STSZ, _abcdk_mp4_atom_read_stsz},
    {ABCDK_MP4_ATOM_TYPE_STCO, _abcdk_mp4_atom_read_stco},
    {ABCDK_MP4_ATOM_TYPE_STSS, _abcdk_mp4_atom_read_stss},
    {ABCDK_MP4_ATOM_TYPE_GMHD, _abcdk_mp4_atom_read_gmhd},
    {ABCDK_MP4_ATOM_TYPE_SMHD, _abcdk_mp4_atom_read_smhd},
    {ABCDK_MP4_ATOM_TYPE_ELST, _abcdk_mp4_atom_read_elst}
};

int abcdk_mp4_atom_read_content(int fd, abcdk_mp4_atom_t *atom)
{
    int (*read_content)(int fd, abcdk_mp4_atom_t *atom) = NULL;
    int chk = -1;

    assert(fd >= 0 && atom != NULL);

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdk_mp4_atom_read_content_methods); i++)
    {
        if (abcdk_mp4_atom_read_content_methods[i].type != atom->type.u32)
            continue;

        read_content = abcdk_mp4_atom_read_content_methods[i].read_content;
        break;
    }

    if (read_content)
        chk = read_content(fd, atom);

    return chk;
}