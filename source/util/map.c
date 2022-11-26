/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/util/map.h"

uint64_t _abcdk_map_hash(const void *data, size_t size, void *opaque)
{
    return abcdk_hash_bkdr64(data, size);
}

int _abcdk_map_compare(const void *key1, size_t size1, const void *key2, size_t size2, void *opaque)
{
    if (size1 > size2)
        return 1;
    else if (size1 < size2)
        return -1;

    return memcmp(key1, key2, size2);
}

void abcdk_map_destroy(abcdk_map_t *map)
{
    assert(map);

    /* 全部释放。*/
    abcdk_tree_free(&map->table);

    memset(map, 0, sizeof(*map));
}

int abcdk_map_init(abcdk_map_t *map, size_t size)
{
    assert(map && size > 0);

    /* 创建树节点，用于表格。 */
    map->table = abcdk_tree_alloc2(NULL, size,0);

    if (!map->table)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, -1);

    /* 如果未指定，则启用默认函数。 */
    if (!map->hash_cb)
        map->hash_cb = _abcdk_map_hash;
    if (!map->compare_cb)
        map->compare_cb = _abcdk_map_compare;

    return 0;
}

abcdk_tree_t *_abcdk_map_find(abcdk_map_t *map, const void *key, size_t ksize, size_t vsize)
{
    abcdk_tree_t *it = NULL;
    abcdk_tree_t *node = NULL;
    uint64_t hash = 0;
    uint64_t bucket = -1;
    int chk = 0;

    assert(map && key && ksize > 0);

    assert(map->table);
    assert(map->hash_cb);
    assert(map->compare_cb);

    hash = map->hash_cb(key, ksize, map->opaque);
    bucket = hash % map->table->obj->numbers;

    /* 查找桶，不存在则创建。*/
    it = (abcdk_tree_t *)map->table->obj->pptrs[bucket];
    if (!it)
    {
        it = abcdk_tree_alloc3(sizeof(bucket));
        if (it)
        {
            /*存放桶的索引值。*/
            ABCDK_PTR2OBJ(uint64_t, it->obj->pptrs[ABCDK_MAP_BUCKET], 0) = bucket;

            /* 桶加入到表格中。*/
            abcdk_tree_insert2(map->table, it, 0);
            map->table->obj->pptrs[bucket] = (uint8_t *)it;
        }
    }

    if (!it)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

    /* 链表存储的节点，依次比较查找。*/
    node = abcdk_tree_child(it, 1);
    while (node)
    {
        chk = map->compare_cb(node->obj->pptrs[ABCDK_MAP_KEY],node->obj->sizes[ABCDK_MAP_KEY], key, ksize, map->opaque);
        if (chk == 0)
            break;

        node = abcdk_tree_sibling(node, 0);
    }

    /*如果节点不存在并且需要创建，则添加到链表头。 */
    if (!node && vsize > 0)
    {
        /*多申请一个字节。*/
        size_t sizes[2] = {ksize + 1, vsize + 1};
        node = abcdk_tree_alloc2(sizes, 2, 0);
        if (!node)
            ABCDK_ERRNO_AND_RETURN1(ENOMEM, NULL);

        /*修正。*/
        node->obj->sizes[ABCDK_MAP_KEY] = ksize;
        node->obj->sizes[ABCDK_MAP_VALUE] = vsize;

        /* 注册数据节点的析构函数。*/
        if (map->destructor_cb)
            abcdk_object_atfree(node->obj, map->destructor_cb, map->opaque);

        /*复制KEY。*/
        memcpy(node->obj->pptrs[ABCDK_MAP_KEY], key, ksize);

        /*也许有构造函数要处理一下。*/
        if(map->construct_cb)
            map->construct_cb(node->obj,map->opaque);

        /* 加入到链表头。 */
        abcdk_tree_insert2(it, node, 1);
    }

    return node;
}

abcdk_object_t *abcdk_map_find(abcdk_map_t *map, const void *key, size_t ksize, size_t vsize)
{
    abcdk_tree_t *node = _abcdk_map_find(map, key, ksize, vsize);

    if (node)
        return node->obj;

    ABCDK_ERRNO_AND_RETURN1(EAGAIN, NULL);
}

void abcdk_map_remove(abcdk_map_t *map, const void *key, size_t ksize)
{
    abcdk_tree_t *node = NULL;

    assert(map);
    assert(map->table && map->hash_cb && map->compare_cb);
    assert(key && ksize > 0);

    node = _abcdk_map_find(map, key, ksize, 0);
    if (node)
    {
        abcdk_tree_unlink(node);
        abcdk_tree_free(&node);
    }
}

int _abcdk_map_scan_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_map_t *map = (abcdk_map_t *)opaque;

    /*已经结束。*/
    if(depth == SIZE_MAX)
        return -1;

    /*跳过组织结构。*/
    if (depth <= 1)
        return 1;

    return map->dump_cb(node->obj, map->opaque);
}

void abcdk_map_scan(abcdk_map_t *map)
{
    assert(map != NULL);
    assert(map->dump_cb != NULL);

    abcdk_tree_iterator_t it = {5,_abcdk_map_scan_cb,(void*)map};

    abcdk_tree_scan(map->table,&it);
}