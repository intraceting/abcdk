/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "util/tree.h"
#include "util/buffer.h"

int dump(size_t deep, abcdk_tree_t *node, void *opaque)
{
    abcdk_tree_fprintf(stderr,deep,node,"%lu\n",node->alloc->sizes[0]);

    // if(deep>=1)
    //     return 0;
    return 1;
}

void traversal(abcdk_tree_t *root)
{
    printf("\n-------------------------------------\n");

    abcdk_tree_iterator_t it = {0,dump,NULL};

    abcdk_tree_scan(root,&it);

 //   abcdk_heap_free2((void **)&it.stack);

    printf("\n-------------------------------------\n");
}

int dump2(size_t deep, abcdk_tree_t *node, void *opaque)
{
    abcdk_tree_fprintf(stderr,deep,node,"%d\n",*ABCDK_PTR2PTR(int,node->alloc->pptrs[0],0));

    // if(deep>=1)
    //     return 0;
    return 1;
}

void traversal2(abcdk_tree_t *root)
{
    printf("\n-------------------------------------\n");

    abcdk_tree_iterator_t it = {0,dump2,NULL};

    abcdk_tree_scan(root,&it);

 //   abcdk_heap_free2((void **)&it.stack);

    printf("\n-------------------------------------\n");
}

void test_tree()
{
    abcdk_tree_t *d = abcdk_tree_alloc(NULL);

    d->alloc = abcdk_object_alloc2(1);

    abcdk_tree_t *n = abcdk_tree_alloc(NULL);

    n->alloc = abcdk_object_alloc2(2);

    abcdk_tree_insert2(d, n,1);

    abcdk_tree_t *n2 = n = abcdk_tree_alloc(NULL);

    n->alloc = abcdk_object_alloc2(3);

    abcdk_tree_insert2(d, n,1);

    n = abcdk_tree_alloc(NULL);

    n->alloc = abcdk_object_alloc2(4);

    abcdk_tree_insert2(d, n,1);

    traversal(d);

    abcdk_tree_t *m = abcdk_tree_alloc(NULL);

    m->alloc = abcdk_object_alloc2(5);

    abcdk_tree_insert2(n, m,0);

    traversal(d);

    m = abcdk_tree_alloc(NULL);

    m->alloc = abcdk_object_alloc2(6);

    abcdk_tree_insert2(n, m,0);

    abcdk_tree_t *m6 = m = abcdk_tree_alloc(NULL);

    m->alloc = abcdk_object_alloc2(7);

    abcdk_tree_insert2(n, m,0);

    abcdk_tree_t *k = abcdk_tree_alloc(NULL);

    k->alloc = abcdk_object_alloc2(8);

    abcdk_tree_insert2(m, k,0);

    k = abcdk_tree_alloc(NULL);

    k->alloc = abcdk_object_alloc2(9);

    abcdk_tree_insert2(m, k,0);

    abcdk_tree_t *u = abcdk_tree_alloc(NULL);

    u->alloc = abcdk_object_alloc2(10);

    abcdk_tree_insert(m, u, k);

    traversal(d);

    abcdk_tree_unlink(m6);

    traversal(d);

    //abcdk_tree_insert_least(d, m6);
    abcdk_tree_insert(d, m6, n2);

    traversal(d);

    abcdk_tree_unlink(m6);
    abcdk_tree_free(&m6);

    traversal(d);


    abcdk_tree_free(&d);
}


int compare_cb(const abcdk_tree_t *node1, const abcdk_tree_t *node2, void *opaque)
{
    int src = *ABCDK_PTR2PTR(int, node1->alloc->pptrs[0], 0);
    int dst = *ABCDK_PTR2PTR(int, node2->alloc->pptrs[0], 0);
    if( src > dst )
        return 1;
    if( src < dst )
        return -1;

    return 0;
}

void test_sort(abcdk_tree_t *t,int by)
{
    abcdk_tree_order_t o = {by,compare_cb,NULL};

    abcdk_tree_sort(t,&o);
}


void test_swap()
{
    abcdk_tree_t *d1 = abcdk_tree_alloc3(sizeof(int));
    *ABCDK_PTR2PTR(int,d1->alloc->pptrs[0],0) = 1;

    abcdk_tree_t *d2 = abcdk_tree_alloc3(sizeof(int));
    *ABCDK_PTR2PTR(int,d2->alloc->pptrs[0],0) = 2;

    abcdk_tree_t *d3 = abcdk_tree_alloc3(sizeof(int));
    *ABCDK_PTR2PTR(int,d3->alloc->pptrs[0],0) = 3;

    abcdk_tree_t *d4 = abcdk_tree_alloc3(sizeof(int));
    *ABCDK_PTR2PTR(int,d4->alloc->pptrs[0],0) = 4;

    abcdk_tree_t *d5 = abcdk_tree_alloc3(sizeof(int));
    *ABCDK_PTR2PTR(int,d5->alloc->pptrs[0],0) = 5;

    abcdk_tree_t *d6 = abcdk_tree_alloc3(sizeof(int));
    *ABCDK_PTR2PTR(int,d6->alloc->pptrs[0],0) = 6;

    abcdk_tree_insert(d1,d2,NULL);
    abcdk_tree_insert(d1,d3,NULL);
    abcdk_tree_insert(d1,d4,NULL);
    abcdk_tree_insert(d1,d5,NULL);
    abcdk_tree_insert(d1,d6,NULL);

    traversal2(d1);

    abcdk_tree_swap(d2,d3);

    traversal2(d1);

    abcdk_tree_swap(d2,d5);

    traversal2(d1);

    abcdk_tree_swap(d6,d2);

    traversal2(d1);

    test_sort(d1,0);

    traversal2(d1);

    test_sort(d1,1);

    traversal2(d1);

    abcdk_tree_free(&d1);
}

int main(int argc, char **argv)
{

    test_tree();

    test_swap();

    return 0;
}