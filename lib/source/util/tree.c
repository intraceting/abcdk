/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/util/tree.h"


abcdk_tree_t *abcdk_tree_father(const abcdk_tree_t *self)
{
    assert(self);

    return self->father;
}

abcdk_tree_t *abcdk_tree_sibling(const abcdk_tree_t *self,int elder)
{
    assert(self);

    if(elder)
        return self->prev;
    
    return self->next;
}


abcdk_tree_t *abcdk_tree_child(const abcdk_tree_t *self,int first)
{
    assert(self);

    if(first)
        return self->first;

    return self->least;
}

void abcdk_tree_unlink(abcdk_tree_t *self)
{
    abcdk_tree_t *root = NULL;

    assert(self);

    /* 获取父节点*/
    root = self->father;

    if (!root)
        return;

    /*
     * 首<--->NODE<--->尾
     * 首<--->NODE
     * NODE<--->尾
     */
    if (self->next)
        self->next->prev = self->prev;
    if (self->prev)
        self->prev->next = self->next;

    /* NODE 是首?*/
    if (self == root->first)
    {
        root->first = self->next;
        if (root->first)
            root->first->prev = NULL;
    }

    /* NODE 是尾? */
    if (self == root->least)
    {
        root->least = self->prev;
        if (root->least)
            root->least->next = NULL;
    }

    /* 打断与父节点的关系链，但同时保留子节点关系链。*/
    self->father = NULL;
    self->next = NULL;
    self->prev = NULL;
}

void abcdk_tree_insert(abcdk_tree_t *father, abcdk_tree_t *child, abcdk_tree_t *where)
{
    assert(father && child);

    /*必须是根节点，或独立节点。 */
    assert(NULL == child->father);
    assert(NULL == child->prev);
    assert(NULL == child->next);

    /* 绑定新父节点。*/
    child->father = father;

    if (where)
    {
        assert(father == where->father);

        if (where == father->first)
        {
            /*添加到头节点之前。*/
            where->prev = child;
            child->next = where;

            /* 新的头节点。*/
            father->first = child;
        }
        else
        {
            /*添加到节点之前*/
            where->prev->next = child;
            child->prev = where->prev;
            child->next = where;
            where->prev = child;
        }
    }
    else
    {
        if (father->least)
        {
            /* 添加到尾节点之后。*/
            father->least->next = child;
            child->prev = father->least;

            /* 新的尾节点。*/
            father->least = child;
        }
        else
        {
            /* 空链表，添加第一个节点。*/
            father->least = father->first = child;
        }
    }
}

void abcdk_tree_insert2(abcdk_tree_t *father, abcdk_tree_t *child,int first)
{
    abcdk_tree_t* where = NULL;

    assert(father && child);

    if(first)
        where = abcdk_tree_child(father,1);

    abcdk_tree_insert(father,child,where);
}

void abcdk_tree_swap(abcdk_tree_t *src,abcdk_tree_t *dst)
{
    abcdk_tree_t *father = NULL;
    abcdk_tree_t *src_next = NULL;
    abcdk_tree_t *dst_next = NULL;

    assert(src && dst);
    assert(src != dst);
    assert(abcdk_tree_father(src) && abcdk_tree_father(dst));
    assert(abcdk_tree_father(src) == abcdk_tree_father(dst));

    father = abcdk_tree_father(src);
    src_next = abcdk_tree_sibling(src,0);
    dst_next = abcdk_tree_sibling(dst,0);

    /* 两个兄弟紧挨着。*/
    if(src_next == dst)
    {
        abcdk_tree_unlink(dst);
        abcdk_tree_insert(father,dst,src);
    }
    else if(dst_next == src)
    {
        abcdk_tree_unlink(src);
        abcdk_tree_insert(father,src,dst);
    }
    else
    {
        /* 有其它兄弟姐妺夹在中间。 */
        abcdk_tree_unlink(dst);
        abcdk_tree_unlink(src);

        if(src_next)
            abcdk_tree_insert(father,dst,src_next);
        else 
            abcdk_tree_insert(father,dst,NULL);
        
        if(dst_next)
            abcdk_tree_insert(father,src,dst_next);
        else
            abcdk_tree_insert(father,src,NULL);
    }
}

void abcdk_tree_free(abcdk_tree_t **root)
{
    abcdk_tree_t *root_p = NULL;
    abcdk_tree_t *father = NULL;
    abcdk_tree_t *node = NULL;
    abcdk_tree_t *child = NULL;

    if(!root || !*root)
        return;

    /* 复制一下 */
    root_p = *root;

    /* 以防清理到父和兄弟节点。 */
    assert(NULL == root_p->father);
    assert(NULL == root_p->prev);
    assert(NULL == root_p->next);

    while (root_p)
    {
        node = abcdk_tree_child(root_p,0);

        if (node)
        {
            child = abcdk_tree_child(node,0);

            /* 检测是否有子节点，如果有先清理子节点。  */
            if (child)
            {
                root_p = node;
            }
            else
            {
                abcdk_tree_unlink(node);

                if(node->destructor_cb)
                    node->destructor_cb(node->obj,node->opaque);

                abcdk_object_unref(&node->obj);
                abcdk_heap_freep((void**)&node);
            }
        }
        else
        {
            /* 没有子节点，返回到父节点。*/
            root_p = abcdk_tree_father(root_p);
        }
    }

    /* 再次复制一下，并清理最野指针。 */
    root_p = *root;
    *root = NULL;

    if (root_p)
    {
        if (root_p->destructor_cb)
            root_p->destructor_cb(root_p->obj, root_p->opaque);

        abcdk_object_unref(&root_p->obj);
        abcdk_heap_free(root_p);
    }
}

abcdk_tree_t *abcdk_tree_alloc(abcdk_object_t *obj)
{
    abcdk_tree_t *node = (abcdk_tree_t *)abcdk_heap_alloc(sizeof(abcdk_tree_t));

    if (!node)
        return NULL;

    node->obj = obj;

    return node;
}

abcdk_tree_t *abcdk_tree_alloc2(size_t *sizes, size_t numbers,int drag)
{
    abcdk_tree_t *node = abcdk_tree_alloc(NULL);
    abcdk_object_t *obj = abcdk_object_alloc(sizes, numbers,drag);

    if (!node || !obj)
        goto final_error;

    node->obj = obj;
    
    return node;

final_error:

    /* 走到这里出错了。 */
    abcdk_tree_free(&node);
    abcdk_object_unref(&obj);
    
    return NULL;
}

abcdk_tree_t *abcdk_tree_alloc3(size_t size)
{
    return abcdk_tree_alloc2(&size,1,0);
}

abcdk_tree_t *abcdk_tree_alloc4(const void *data, size_t size)
{
    abcdk_tree_t *p = NULL;

    assert(data != NULL && size > 0);

    p = abcdk_tree_alloc3(size+1);
    if(!p)
        return NULL;

    memcpy(p->obj->pptrs[0],data,size);
    p->obj->sizes[0] = size;

    return p;
}

void abcdk_tree_scan(abcdk_tree_t *root,abcdk_tree_iterator_t* it)
{
    abcdk_tree_t *node = NULL;
    abcdk_tree_t *child = NULL;
    abcdk_tree_t **stack = NULL;
    size_t stack_size = PATH_MAX/2;
    size_t depth = 0;// begin 0
    int chk;

    assert(root != NULL && it != NULL);
    assert(it->dump_cb!= NULL);
        
    /* 如果调用者不确定，则在内部自动确定。  */
    if (it->depth_max > 0)
        stack_size = it->depth_max;

    stack = (abcdk_tree_t **)abcdk_heap_alloc(stack_size * sizeof(abcdk_tree_t *));
    if (!stack)
        return;

    /*根*/
    chk = it->dump_cb(0,root,it->opaque);
    if(chk <= 0)
        goto final;

    /* 从第一个孩子开始遍历。 */
    node = abcdk_tree_child(root,1);

    while(node)
    {
        chk = it->dump_cb(depth + 1, node, it->opaque);
        if (chk < 0)
            goto final;

        if(chk > 0)
            child = abcdk_tree_child(node,1);
        else 
            child = NULL;

        if(child)
        {
            assert(stack_size > depth);

            stack[depth++] = node;

            node = child;
        }
        else
        {
            node = abcdk_tree_sibling(node,0);

            while (!node && depth > 0)
            {
                node = stack[--depth];
                node = abcdk_tree_sibling(node,0);
            }
        }
    }

final:

    it->dump_cb(SIZE_MAX, NULL, it->opaque);

    abcdk_heap_freep((void**)&stack);
}

void abcdk_tree_sort(abcdk_tree_t *father,abcdk_tree_iterator_t *it,int order)
{
    abcdk_tree_t *t1 = NULL;
    abcdk_tree_t *t2 = NULL;
    abcdk_tree_t *t3 = NULL;
    int chk;

    assert(father != NULL && it != NULL);
    assert(it->compare_cb != NULL);

    t2 = abcdk_tree_child(father, 1);
    while (t2)
    {
        t3 = abcdk_tree_sibling(t1 = t2, 0);
        while (t3)
        {
            chk = it->compare_cb(t1,t3,it->opaque);

            if(order)
            {
                if (chk > 0)
                    t1 = t3;
            }
            else
            {
                if (chk < 0)
                    t1 = t3;
            }

            t3 = abcdk_tree_sibling(t3, 0);
        }

        /*需要交换时，再进行交换。*/
        if (t1 != t2)
            abcdk_tree_swap(t1, t2);

        t2 = abcdk_tree_sibling(t1, 0);
    }
}

void abcdk_tree_distinct(abcdk_tree_t *father,abcdk_tree_iterator_t *it)
{
    abcdk_tree_t *t1 = NULL;
    abcdk_tree_t *t2 = NULL;
    abcdk_tree_t *t3 = NULL;
    int chk;

    assert(father != NULL && it != NULL);
    assert(it->compare_cb != NULL);

    t2 = abcdk_tree_child(father, 1);
    if(!t2)
        return;

    for (;;)
    {
        t3 = abcdk_tree_sibling(t2, 0);
        if (!t3)
            return;

        chk = it->compare_cb(t2, t3, it->opaque);

        if (chk != 0)
            t2 = t3;
        else 
        {
            abcdk_tree_unlink(t3);
            abcdk_tree_free(&t3);
        }
    }
}


ssize_t abcdk_tree_fprintf(FILE* fp,size_t depth,const abcdk_tree_t *node,const char* fmt,...)
{
    ssize_t wsize = 0;

    va_list vaptr;
    va_start(vaptr, fmt);

    wsize = abcdk_tree_vfprintf(fp,depth,node,fmt,vaptr);

    va_end(vaptr);

    return wsize;
}

ssize_t abcdk_tree_vfprintf(FILE* fp,size_t depth,const abcdk_tree_t *node,const char* fmt,va_list args)
{
    abcdk_tree_t *tmp = NULL;
    abcdk_tree_t **stack = NULL;
    ssize_t wsize = 0;
    ssize_t wsize2 = 0;

    assert(fp && node && fmt);

    if (depth <= 0)
    {
        wsize2 = vfprintf(fp,fmt,args);
        if (wsize2 <= 0)
            goto final;

        wsize += wsize2;
    }
    else
    {
        /*准备堆栈。 */
        stack = abcdk_heap_alloc(depth * sizeof(abcdk_tree_t *));
        if(!stack)
            ABCDK_ERRNO_AND_RETURN1(ENOMEM,-1);

        tmp = (abcdk_tree_t *)node;

        for (size_t i = 1; i < depth; i++)
            stack[depth-i] = (tmp = abcdk_tree_father(tmp));

        for (size_t i = 1; i < depth; i++)
        {
            if (abcdk_tree_sibling((abcdk_tree_t *)stack[i], 0))
                wsize2 = fprintf(fp, "│   ");
            else
                wsize2 = fprintf(fp, "    ");

            if (wsize2 <= 0)
                goto final;

            wsize += wsize2;
        }


        if (abcdk_tree_sibling(node, 0))
            wsize2 = fprintf(fp, "├── ");
        else
            wsize2 = fprintf(fp, "└── ");

        if (wsize2 <= 0)
            goto final;

        wsize += wsize2;

        wsize2 = vfprintf(fp,fmt,args);

        if (wsize2 <= 0)
            goto final;

        wsize += wsize2;
    }

final:

    abcdk_heap_freep((void**)&stack);

    return wsize;
}

ssize_t abcdk_tree_snprintf(char *buf, size_t max, size_t depth, const abcdk_tree_t *node, const char *fmt, ...)
{
    ssize_t wsize = 0;

    assert(buf != NULL && max >0 && node != NULL && fmt != NULL);

    va_list vaptr;
    va_start(vaptr, fmt);

    wsize = abcdk_tree_vsnprintf(buf,max,depth,node,fmt,vaptr);

    va_end(vaptr);
    
    return wsize;
}

ssize_t abcdk_tree_vsnprintf(char *buf, size_t max, size_t depth, const abcdk_tree_t *node,const char* fmt,va_list args)
{
    FILE* fp = NULL;
    ssize_t wsize = 0;

    assert(buf != NULL && max >0 && node != NULL && fmt != NULL);

    fp = fmemopen(buf,max,"w");
    if(!fp)
        return -1;

    wsize = abcdk_tree_vfprintf(fp,depth,node,fmt,args);

    fclose(fp);
    
    return wsize;
}
