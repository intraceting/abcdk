/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_MAP_H
#define ABCDK_UTIL_MAP_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/hash.h"

__BEGIN_DECLS

/**
 * 简单的MAP容器.
 * 
 * HASH和DEQUE存储结构.
*/
typedef struct _abcdk_map
{
    /**
     * 表格.
     * 
     * @note 尽量不要直接修改.
    */
    abcdk_tree_t *table;

    /**
     * 环境指针.
    */
    void *opaque;
    
    /**
     * KEY哈希函数.
    */
    uint64_t (*hash_cb)(const void* key,size_t size,void *opaque);

    /**
     * KEY比较函数.
     * 
     * @return 0 key1 == key2, !0 key1 != key2.
    */
    int (*compare_cb)(const void *key1, size_t size1, const void *key2,size_t size2,void *opaque);

    /** 
     * 构造回调函数.
    */
    void (*construct_cb)(abcdk_object_t *obj, void *opaque);

    /**
     * 析构回调函数.
    */
    void (*destructor_cb)(abcdk_object_t *obj, void *opaque);

    /**
     * 删除回调函数.
    */
    void (*remove_cb)(abcdk_object_t *obj, void *opaque);

    /**
     * 回显回调函数.
     * 
     * @return 1 继续, -1 终止.
    */
    int (*dump_cb)(abcdk_object_t *obj, void *opaque);

    
}abcdk_map_t;

/**
 * MAP的字段索引.
*/
typedef enum _abcdk_map_field
{
    /**
     * Bucket.
    */
   ABCDK_MAP_BUCKET = 0,
#define ABCDK_MAP_BUCKET     ABCDK_MAP_BUCKET

    /**
     * Key.
    */
   ABCDK_MAP_KEY = 0,
#define ABCDK_MAP_KEY        ABCDK_MAP_KEY

    /**
     * Value.
    */
   ABCDK_MAP_VALUE = 1
#define ABCDK_MAP_VALUE      ABCDK_MAP_VALUE

}abcdk_map_field_t;


/** 释放.*/
void abcdk_map_destroy(abcdk_map_t **map);

/** 创建.*/
abcdk_map_t *abcdk_map_create(size_t size);

/**
 * 查找或创建节点.
 * 
 * @param key Key指针.
 * @param ksize Key长度.
 * @param vsize Value长度. 0 仅查找, > 0 不存在则创建.
 * 
 * @return !NULL(0) 节点指针(复制的指针, 不需要主动释放), NULL(0) 不存在或创建失败.
 * 
*/
abcdk_object_t* abcdk_map_find(abcdk_map_t* map,const void* key,size_t ksize,size_t vsize);
#define abcdk_map_find2(map,key,vsize) abcdk_map_find((map),(key),sizeof(*(key)),(vsize))

/**
 * 删除.
*/
void abcdk_map_remove(abcdk_map_t* map,const void* key,size_t ksize);
#define abcdk_map_remove2(map,key) abcdk_map_remove((map),(key),sizeof(*(key)))

/**
 * 扫描节点.
 * 
 * @note 深度优先.
*/
void abcdk_map_scan(abcdk_map_t *map);


__END_DECLS

#endif //ABCDK_UTIL_MAP_H