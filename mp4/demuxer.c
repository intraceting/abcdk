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

int _abcdk_mp4_read_probe_stbl(abcdk_tree_t *root, int fd)
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
        // if (atom->type.u32 == )
        // {
        //     /*跳转文件指针到容器内部。*/
        //     lseek(fd, atom->off_cont, SEEK_SET);
        // }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_minf(abcdk_tree_t *root, int fd)
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
        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STBL)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_stbl(node,fd);
            if (chk != 0)
                goto final_error;
        }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}


int _abcdk_mp4_read_probe_edts(abcdk_tree_t *root, int fd)
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
        // if (atom->type.u32 == )
        // {
        //     /*跳转文件指针到容器内部。*/
        //     lseek(fd, atom->off_cont, SEEK_SET);
        // }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_mdia(abcdk_tree_t *root, int fd)
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
        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MINF)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_minf(node,fd);
            if (chk != 0)
                goto final_error;
        }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_trak(abcdk_tree_t *root, int fd)
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
        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_EDTS)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_edts(node,fd);
            if (chk != 0)
                goto final_error;
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MDIA)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_mdia(node,fd);
            if (chk != 0)
                goto final_error;
        }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_mvex(abcdk_tree_t *root, int fd)
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
        // if (atom->type.u32 == )
        // {
        //     /*跳转文件指针到容器内部。*/
        //     lseek(fd, atom->off_cont, SEEK_SET);
        // }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_moov(abcdk_tree_t *root, int fd)
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
        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TRAK)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_trak(node, fd);
            if (chk != 0)
                goto final_error;
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MVEX)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_mvex(node, fd);
            if (chk != 0)
                goto final_error;
        }
 
        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_traf(abcdk_tree_t *root, int fd)
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
        // if (atom->type.u32 == )
        // {
        //     /*跳转文件指针到容器内部。*/
        //     lseek(fd, atom->off_cont, SEEK_SET);
        // }
 
        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_moof(abcdk_tree_t *root, int fd)
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
        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TRAF)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_traf(node, fd);
            if (chk != 0)
                goto final_error;
        }
 
        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_mfra(abcdk_tree_t *root, int fd)
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
        // if (atom->type.u32 == )
        // {
        //     /*跳转文件指针到容器内部。*/
        //     lseek(fd, atom->off_cont, SEEK_SET);
        // }
 
        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

int _abcdk_mp4_read_probe_skip(abcdk_tree_t *root, int fd)
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
        // if (atom->type.u32 == )
        // {
        //     /*跳转文件指针到容器内部。*/
        //     lseek(fd, atom->off_cont, SEEK_SET);
        // }
 
        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);

        /*限制在容器内部解析。*/
        keep = ((atom->off_head + atom->size< root_atom->off_head+root_atom->size)?1:0);
    }

    return 0;

final_error:

    return -1;
}

abcdk_tree_t *abcdk_mp4_read_probe(int fd)
{
    abcdk_mp4_atom_t *atom = NULL;
    abcdk_tree_t *root;
    abcdk_tree_t *node = NULL;
    int keep = 1;
    int chk;

    root = abcdk_tree_alloc3(1);
    if (!root)
        goto final;

    while (keep)
    {
        node = _abcdk_mp4_read_atom_header(fd);
        if (!node)
            goto final;

        abcdk_tree_insert2(root, node, 0);

        atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MOOV)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_moov(node, fd);
            if (chk != 0)
                goto final;

            /*stop*/
            keep = 0;
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MOOF)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_moof(node, fd);
            if (chk != 0)
                goto final;
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MFRA)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_mfra(node, fd);
            if (chk != 0)
                goto final;
        }
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_SKIP)
        {
            /*跳转文件指针到容器内部。*/
            lseek(fd, atom->off_cont, SEEK_SET);

            chk = _abcdk_mp4_read_probe_skip(node, fd);
            if (chk != 0)
                goto final;
        }

        /*跳转文件指针到下一个atom。*/
        lseek(fd, atom->off_head + atom->size, SEEK_SET);
    }

final:

    return root;
}