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
#include <locale.h>
#include "entry.h"

int abcdk_test_any(abcdk_option_t *args)
{
#if 0
    size_t b = 100;
    int a = ABCDK_CLAMP(0,-(ssize_t)b,(ssize_t)b);

    //int a = ABCDK_MAX((ssize_t)c,(ssize_t)b);

    printf("a=%d\n",a);
#elif 0

    char url[]={"http://asdfasfdasdf.asdfasdf.asdfasdfasf/a/////b/cccc/../ccccc/./././eeeee/?dsdfadf=bbbb&ddsafsd=rrrr#dddd"};

    abcdk_url_abspath(url,0);

    printf("%s\n",url);

    abcdk_object_t *p=NULL;


    p = abcdk_url_fixpath("/bbbb","http://aaaa.com");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","http://aaaa.com/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","http://aaaa.com");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","http://aaaa.com/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","http://aaaa.com/cccc");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","http://aaaa.com/cccc");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("../bbbb","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("../../bbbb","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);


    p = abcdk_url_fixpath("../../bbbb/../","http://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("://bbbb","https://aaaa.com/cccc/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);


    p = abcdk_url_fixpath("http://bbbb/?ddddd","http://aaaa.com/cccc?ddsdssdd=ddd&ewerqer=rrrr#ffsfg");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("http://bbbb","/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);


    p = abcdk_url_fixpath("/bbbb","/aaaa");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","aaaa");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("/bbbb","aaaa/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","/aaaa");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","aaaa");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);

    p = abcdk_url_fixpath("bbbb","aaaa/");
    printf("%s\n", p->pstrs[0]);
    abcdk_object_unref(&p);
#elif 0

    abcdk_object_t *p = NULL;

    p = abcdk_url_split("/aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("/aaa.aaa/bbb?");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("/aaa.aaa/bbb?aaadd=bbbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("/aaa.aaa/bbb?aaadd=bbbb#");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("/aaa.aaa/bbb?aaadd=bbbb#ssss");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("//aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("://aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i,p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa/");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("file:///aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://aaa.aaa:80/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://[22::1]:80/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://user:password@aaa.aaa/bbb");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://user:password@aaa.aaa:80/bbb?asfsfd=bbbb&cccc=dddd");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://user:password@[22::1]:80/bbb?");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("http://user@aaa.aaa/bbb?asfsfd=bbbb&cccc=dddd#");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);

    p = abcdk_url_split("://user@aaa.aaa/bbb?asfsfd=bbbb&cccc=dddd#aaaa");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
    
    p = abcdk_url_split("://user@aaa.aaa/bbb#aaaa");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
    
    p = abcdk_url_split("://user@aaa.aaa/#aaaa");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
        
    p = abcdk_url_split("://user@aaa.aaa#又c");
    for (int i = 0; i < p->numbers; i++)
        printf("[%d]={%s}\n",i, p->pstrs[i]);
    abcdk_object_unref(&p);
#elif 0

    uint32_t a = 0x00123456;

    printf("%08x\n",a);
    printf("%08x\n",abcdk_endian_h_to_b32(a));

#elif 0


    /*
        1,2,3,4,5,6,7,8,9,0
        a,s,d,f,g,h,j,k,l,;
        q,w,e,r,t,y,u,i,o,p
    */
    int len = 6;
    char src[6] ={"abcdef"};
    char dst[6] = {0};
    char dst2[6] = {0};

    uint8_t wheels[4][2][256];

    uint8_t pool[256 / 8] ={0},pool2[256 / 8] ={0},pool3[256 / 8] ={0},pool4[256 / 8] ={0};
    size_t size = 256 / 8;

    uint64_t seed = 3;

    for (int i = 0; i < 256; i++)
    {
        for(;;)
        {
            int c = abcdk_rand(&seed)%256;
            int chk = abcdk_bloom_mark(pool,size,c);
            if(chk)
                continue;

            wheels[0][0][i] = (c)%256;
            break;
        }

        
        wheels[0][1][wheels[0][0][i]] = i;
        

        for(;;)
        {
            int c = abcdk_rand(&seed)%256;
            int chk = abcdk_bloom_mark(pool2,size,c);
            if(chk)
                continue;

            wheels[1][0][i] = (c)%256;
            break;
        }
        
        wheels[1][1][wheels[1][0][i]] = i;

         for(;;)
        {
            int c = abcdk_rand(&seed)%256;
            int chk = abcdk_bloom_mark(pool3,size,c);
            if(chk)
                continue;

            wheels[2][0][i] = (c)%256;
            break;
        }

        wheels[2][1][wheels[2][0][i]] = i;

           for(;;)
        {
            int c = abcdk_rand(&seed)%256;
            int chk = abcdk_bloom_mark(pool4,size,c);
            if(chk)
                continue;

            wheels[3][0][i] = (c)%256;
            break;
        }

        wheels[3][1][wheels[3][0][i]] = i;
    }

    for (int i3 = 0; i3 < 3; i3++)
    for (int i2 = 0; i2 < 4; i2++)
    for (int i1 = 0; i1 < 5; i1++)
    for (int i0 = 0; i0 < 256; i0++)
    {
        int s = 'a';
        uint8_t c = s;
        c = wheels[0][0][(c+i0)%256];
        c = wheels[1][0][(c+i1)%256];
        c = wheels[2][0][(c+i2)%256];
        c = wheels[3][0][(c+i3)%256];
        c = (~c) % 256;
        c = wheels[3][1][c] - i3;
        c = wheels[2][1][c] - i2;
        c = wheels[1][1][c] - i1;
        c = wheels[0][1][c] - i0;

        //printf("s = %d, c=%hhd\n",s,c);

        int m = c;

        c = wheels[0][0][(c+i0)%256];
        c = wheels[1][0][(c+i1)%256];
        c = wheels[2][0][(c+i2)%256];
        c = wheels[3][0][(c+i3)%256];
        c = (~c) % 256;
        c = wheels[3][1][c] - i3;
        c = wheels[2][1][c] - i2;
        c = wheels[1][1][c] - i1;
        c = wheels[0][1][c] - i0;

        printf("[%d,%d,%d,%d] s=%d,m=%d,c=%hhd\n",i0,i1,i2,i3,s,m, c);
    }
    
    for(int i =0;i<len;i++)
        printf("|%02hhx|%02hhx|%02hhx|\n",src[i],dst[i],dst2[i]);
#elif 0

    size_t rows = 3;
    size_t cols = 256;

    uint8_t *send_dist = abcdk_heap_alloc(rows * cols);
    uint8_t *recv_dist = abcdk_heap_alloc(rows * cols);

    uint64_t send_seed = 1 ,recv_seed = 1;

    abcdk_enigma_mkdict(&send_seed, send_dist, rows, cols);
    abcdk_enigma_mkdict(&recv_seed, recv_dist, rows, cols);

    abcdk_enigma_t *send_ctx = abcdk_enigma_create(send_dist, rows, cols);
    abcdk_enigma_t *recv_ctx = abcdk_enigma_create(recv_dist, rows, cols);

    int n = 1;
    char a[] = {"aaaaaa"};
    char b[100] = {0},c[100] = {0};
    
    uint64_t st=0;
    abcdk_clock(st,&st);

    for(int i=0;i<100000000;i++)
    {
        abcdk_enigma_light_batch(send_ctx, b, a, n);
    //    abcdk_enigma_light_batch(recv_ctx, c, b, n);
    }

    printf("cast:%lu\n",abcdk_clock(st,&st));

    abcdk_enigma_light_batch(send_ctx, b, a, n);
    abcdk_enigma_light_batch(recv_ctx, c, b, n);

    abcdk_enigma_free(&send_ctx);
    abcdk_enigma_free(&recv_ctx);

    abcdk_heap_free(send_dist);
    abcdk_heap_free(recv_dist);
#elif 0

    size_t rows = 3;
    size_t cols = 256;

    uint8_t *send_dist = abcdk_heap_alloc(rows * cols);
    uint8_t *recv_dist = abcdk_heap_alloc(rows * cols);

    uint64_t send_seed = 1234 ,recv_seed = 1234;

    abcdk_enigma_mkdict(&send_seed, send_dist, rows, cols);
    abcdk_enigma_mkdict(&recv_seed, recv_dist, rows, cols);

    abcdk_enigma_t *send_ctx = abcdk_enigma_create(send_dist, rows, cols);
    abcdk_enigma_t *recv_ctx = abcdk_enigma_create(recv_dist, rows, cols);

    abcdk_object_t *f = abcdk_mmap_filename("./test1.mp4",0,0,0);

    uint64_t st=0;
    abcdk_clock(st,&st);
    
    for(size_t i =0;i<f->sizes[0];i++)
    {
        uint8_t s = f->pptrs[0][i];
        uint8_t d = abcdk_enigma_light(send_ctx,s);
        uint8_t d2 = abcdk_enigma_light(recv_ctx,d);
        assert(s == d2);
    }

    printf("cast:%lu\n",abcdk_clock(st,&st));

    abcdk_object_unref(&f);

    abcdk_enigma_free(&send_ctx);
    abcdk_enigma_free(&recv_ctx);

    abcdk_heap_free(send_dist);
    abcdk_heap_free(recv_dist);

#elif 1

    for (int y = 3; y <= 256; y++)
    {
        #pragma omp parallel for num_threads(4)
        for (int x = 4; x <= 256; x += 2)
        {
            printf("row=%d,col=%d\n",y,x);

            size_t rows = y;
            size_t cols = x;

            uint8_t *dist = abcdk_heap_alloc(rows * cols);

            uint64_t seed = x;

            abcdk_enigma_mkdict(&seed, dist, rows, cols);

            abcdk_enigma_t *send_ctx = abcdk_enigma_create(dist, rows, cols);
            abcdk_enigma_t *recv_ctx = abcdk_enigma_create(dist, rows, cols);

            for (int i = 0; i < 1000; i++)
            {
                int n = rand() % 500 + 1;
                char src[600] = {0};
                char dst[600] = {0};
                char dst2[600] = {0};

                for (int j = 0; j < n; j++)
                    src[j] = rand() % cols;

                abcdk_enigma_light_batch(send_ctx, dst, src, n);
                //     printf("---------------------------\n");
                abcdk_enigma_light_batch(recv_ctx, dst2, dst, n);

                int chk = memcmp(src, dst2, n);
                assert(chk == 0);
            }

            abcdk_enigma_free(&send_ctx);
            abcdk_enigma_free(&recv_ctx);

            abcdk_heap_free(dist);
        }
    }


#endif 
}