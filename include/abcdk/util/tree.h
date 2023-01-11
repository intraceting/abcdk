/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_TREE_H
#define ABCDK_UTIL_TREE_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"

__BEGIN_DECLS

/**
 * 树节点。
 * 
 * 比较原始的树节点结构。
*/
typedef struct _abcdk_tree
{
    /** 父。*/
    struct _abcdk_tree *father;

    /** 大娃。*/
    struct _abcdk_tree *first;

    /** 么娃。*/
    struct _abcdk_tree *least;

    /** 兄长。*/
    struct _abcdk_tree *prev;

    /** 小弟。*/
    struct _abcdk_tree *next;

    /**
     * 数据对象。
     * 
     * @note 当节点被删除时，自动调用abcdk_object_unref()释放。
    */
    abcdk_object_t *obj;

}abcdk_tree_t;

/**
 * 树节点迭代器。
*/
typedef struct _abcdk_tree_iterator
{
    /** 
     * 最大深度。
     * 
     * 如果无法确定填多少合适，就填0。
    */
    size_t depth_max;

    /**
     * 回显函数。
     * 
     * @return -1 终止，0 忽略孩子，1 继续。
    */
    int (*dump_cb)(size_t depth, abcdk_tree_t *node, void *opaque);

    /**
     * 环境指针。
    */
    void *opaque;

} abcdk_tree_iterator_t;

/**
 * 树节点排序规则。
 * 
*/
typedef struct _abcdk_tree_order
{
    /**
     * 顺序。!0 升序，0 降序。
    */
    int by;

    /**
     * 比较函数。
     * 
     * @return > 0 is node1 > node2，0 is node1 == node2，< 0 is node1 < node2。
    */
    int (*compare_cb)(const abcdk_tree_t *node1, const abcdk_tree_t *node2, void *opaque);

    /**
     * 环境指针。
    */
    void *opaque;

} abcdk_tree_order_t;

/**
 * 获取自己的父节指针。
*/
abcdk_tree_t *abcdk_tree_father(const abcdk_tree_t *self);

/**
 * 获取自己的兄弟指针。
 * 
 * @param elder 0 找小弟，!0 找兄长。
*/
abcdk_tree_t *abcdk_tree_sibling(const abcdk_tree_t *self,int elder);

/**
 * 获取自己的孩子指针。
 * 
 * @param first 0 找么娃，!0 找大娃。
*/
abcdk_tree_t *abcdk_tree_child(const abcdk_tree_t *self,int first);

/**
 * 断开自己在树中的关系链。
 * 
 * @note 自己的孩子，以孩子的孩子都会跟随自己从树节点中断开。
 * 
*/
void abcdk_tree_unlink(abcdk_tree_t *self);

/**
 * 插入节点到树的关系链中。
 * 
 * @param father 父。
 * @param child 孩子。
 * @param where NULL(0) 孩子为小弟，!NULL(0) 孩子为兄长。
 * 
*/
void abcdk_tree_insert(abcdk_tree_t *father, abcdk_tree_t *child, abcdk_tree_t *where);

/**
 * 插入节点到树的关系链中。
 * 
 * @param father 父。
 * @param child 孩子。
 * @param first 0 孩子为么娃，!0 孩子为大娃。
 * 
*/
void abcdk_tree_insert2(abcdk_tree_t *father, abcdk_tree_t *child,int first); 

/**
 * 节点交换。
 * 
 * @note 必须是同一个父节点的子节点。
 * 
*/
void abcdk_tree_swap(abcdk_tree_t *src,abcdk_tree_t *dst);

/**
 * 删除节点。
 * 
 * @note 包括自己，自己的孩子，以孩子的孩子都会被删除。
 * 
 * @param root 节点指针的指针。当接口返回时，被赋值NULL(0)。
*/
void abcdk_tree_free(abcdk_tree_t **root);

/**
 * 创建节点。
 * 
 * @param alloc 内存块指针，可以为NULL(0)。注：仅复制指针，不会改变对象的引用计数。
 * 
*/
abcdk_tree_t *abcdk_tree_alloc(abcdk_object_t *alloc);

/**
 * 创建节点，同时申请数据内存块。
*/
abcdk_tree_t *abcdk_tree_alloc2(size_t *sizes,size_t numbers,int drag);

/**
 * 创建节点，同时申请数据内存块。
*/
abcdk_tree_t *abcdk_tree_alloc3(size_t size);

/**
 * 扫描树节点。
 * 
 * @note 深度优先遍历节点。
*/
void abcdk_tree_scan(abcdk_tree_t *root,abcdk_tree_iterator_t* it);

/** 
 * 排序。
 * 
 * @note 选择法排序。只排序子节点，如需要对整颗树排序，需要接合迭代器。
 * 
*/
void abcdk_tree_sort(abcdk_tree_t *father,abcdk_tree_order_t *order);

/**
 * 格式化打印。
 * 
 * @note 输出有层次感的树形图。
 * @note 不会在末尾添加'\n'(换行)字符。
 * 
 * @return >=0 成功(输出的长度)，< 0 失败。
*/
ssize_t abcdk_tree_fprintf(FILE* fp,size_t depth,const abcdk_tree_t *node,const char* fmt,...);

/**
 * 格式化打印。
 * 
 * @see abcdk_tree_fprintf
 * 
 * @return >=0 成功(输出的长度)，< 0 失败。
*/
ssize_t abcdk_tree_vfprintf(FILE* fp,size_t depth,const abcdk_tree_t *node,const char* fmt,va_list args);

/**
 * 格式化打印。
 * 
 * @see abcdk_tree_fprintf
 * 
 * @return >=0 输出的长度，< 0 失败。
*/
ssize_t abcdk_tree_snprintf(char *buf, size_t max,size_t depth, const abcdk_tree_t *node, const char *fmt, ...);

/**
 * 格式化打印。
 * 
 * @see abcdk_tree_fprintf
 * 
 * @return >=0 输出的长度，< 0 失败。
*/
ssize_t abcdk_tree_vsnprintf(char *buf, size_t max,size_t depth, const abcdk_tree_t *node,const char* fmt,va_list args);


__END_DECLS

#endif //ABCDK_UTIL_TREE_H