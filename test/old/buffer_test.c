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
#include "abcdk-util/buffer.h"
#include "abcdk-util/pool.h"

void test1()
{
    abcdk_buffer_t *a = abcdk_buffer_alloc2(1000);

   abcdk_buffer_printf(a,"12345678");

   abcdk_buffer_t *b = abcdk_buffer_copy(a);

   abcdk_buffer_printf(b,"aaaaaa");

    printf("a={%s}\n",(char*)a->data);
    printf("b={%s}\n",(char*)b->data);

   abcdk_buffer_free(&a);
   abcdk_buffer_free(&b);

   void* c = abcdk_heap_alloc(1000);

   abcdk_buffer_t *d = abcdk_heap_alloc(sizeof(abcdk_buffer_t));

    d->data = c;
    d->size = 1000;

    abcdk_buffer_printf(d,"aaaaaa");

    printf("d={%s}\n",(char*)d->data);

    abcdk_buffer_t *d2 =abcdk_buffer_copy(d);

    abcdk_heap_free2((void**)&d);
    abcdk_heap_free2(&c);
    abcdk_buffer_free(&d2);

    abcdk_buffer_t *e = abcdk_buffer_alloc2(1000*100000);

    abcdk_buffer_t *f = abcdk_buffer_copy(e);
    abcdk_buffer_t *g = abcdk_buffer_copy(f);

    abcdk_buffer_t *h = abcdk_buffer_clone(g);

    abcdk_buffer_free(&e);
    abcdk_buffer_free(&f);

    abcdk_buffer_free(&g);
    abcdk_buffer_free(&h);
}

void test2()
{
    abcdk_buffer_t *a = abcdk_buffer_alloc2(10);

    printf("%lu\n",abcdk_buffer_write(a,"aaaaaa",6));

    printf("%lu\n",abcdk_buffer_printf(a,"%s","bb"));

    printf("\r%lu\n",abcdk_buffer_fill(a,'c'));

    printf("%lu\n" ,abcdk_buffer_write(a,"dddd",6));

    char buf[11] = {0};

    fprintf(stderr,"%lu\n",abcdk_buffer_read(a,buf,7));

    abcdk_buffer_drain(a);

    fprintf(stderr,"%lu\n",abcdk_buffer_read(a,buf,7));

    fprintf(stderr,"%lu\n",abcdk_buffer_write(a,"abcdefg",6));

    abcdk_buffer_resize(a,1000);

    abcdk_buffer_free(&a);
}

void test3()
{
    abcdk_buffer_t *a = abcdk_buffer_alloc(NULL);

    abcdk_buffer_t *b = abcdk_buffer_copy(a);

    abcdk_buffer_t *c = abcdk_buffer_clone(b);

    abcdk_buffer_free(&a);
    abcdk_buffer_free(&b);
    abcdk_buffer_free(&c);
}

void test4()
{
    abcdk_pool_t p = {0};

    abcdk_pool_init(&p,sizeof(size_t),3);

    printf("\n---------------\n");

    size_t id = -1;
    for(int i = 0;i<3;i++)
    {
        assert(abcdk_pool_pull(&p,&id,sizeof(id))<=0);
        printf("%ld\n",id);
    }

    printf("\n---------------\n");

    assert(abcdk_pool_pull(&p,&id,sizeof(id)) == -1);

    id = 2;
    assert(abcdk_pool_push(&p,&id,sizeof(id))>0);
    id = 1;
    assert(abcdk_pool_push(&p,&id,sizeof(id))>0);
    id = 3;
    assert(abcdk_pool_push(&p,&id,sizeof(id))>0);
 //   assert(abcdk_pool_push(&p,&id,sizeof(id))>0);
    

    printf("\n---------------\n");

    for(int i = 0;i<p.table->numbers;i++)
    {
         assert(abcdk_pool_pull(&p,&id,sizeof(id))>0);
        printf("%ld\n",id);

        assert(abcdk_pool_push(&p,&id,sizeof(id))>0);
    }

    printf("\n---------------\n");

    abcdk_pool_destroy(&p);
}

int main(int argc, char **argv)
{
   
   test1();

   test2();

    test3();

    test4();

   return 0;
}