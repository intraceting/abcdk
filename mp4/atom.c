/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-mp4/atom.h"

int _abcdk_mp4_find_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    if (depth == -1UL)
        return -1;

    abcdk_allocator_t *ctx = (abcdk_allocator_t *)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];

    if (atom->type.u32 != ABCDK_PTR2PTR(abcdk_mp4_tag_t, ctx->pptrs[0], 0)->u32)
        return 1;

    /*走到这里时，已经找到。*/
    ctx->pptrs[1] = (uint8_t*)node;

    return -1;
}

abcdk_tree_t *abcdk_mp4_find(abcdk_tree_t *root, abcdk_mp4_tag_t *type)
{
    abcdk_allocator_t *ctx = NULL;
    abcdk_tree_t *atom = NULL;

    assert(root != NULL && type != NULL);

    size_t sizes[2] = {sizeof(abcdk_mp4_tag_t), 0};
    ctx = abcdk_allocator_alloc(sizes, 2, 0);
    if (!ctx)
        return NULL;

    ABCDK_PTR2PTR(abcdk_mp4_tag_t, ctx->pptrs[0], 0)->u32 = type->u32;

    abcdk_tree_iterator_t it = {0, _abcdk_mp4_find_cb, ctx};
    abcdk_tree_scan(root, &it);

    atom = (abcdk_tree_t *)ctx->pptrs[1];

    abcdk_allocator_unref(&ctx);

    return atom;
}

abcdk_tree_t *abcdk_mp4_find2(abcdk_tree_t *root,uint32_t type)
{
    abcdk_mp4_tag_t tag = {0};

    assert(root != NULL && type != 0);

    tag.u32 = type;

    return abcdk_mp4_find(root,&tag);
}