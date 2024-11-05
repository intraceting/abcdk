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

#ifdef HAVE_PAM
#include <security/pam_appl.h>
#endif //

#ifdef HAVE_NCURSES
#include <ncurses.h>
#endif //


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

#elif 0

    for (int y = 2; y <= 128; y++)
    {
   //     #pragma omp parallel for num_threads(8)
        for (int x = 4; x <= 256; x += 2)
        {
            printf("row=%d,col=%d\n",y,x);

            abcdk_enigma_t *send_ctx = abcdk_enigma_create(y, x);
            abcdk_enigma_t *recv_ctx = abcdk_enigma_create(y, x);

            abcdk_enigma_init_ex(send_ctx,"abcd",4);
            abcdk_enigma_init_ex(recv_ctx,"abcd",4);

            for (int i = 0; i < 10; i++)
            {
                int n = rand() % 500 + 1;
                uint8_t src[600] = {0};
                uint8_t dst[600] = {0};
                uint8_t dst2[600] = {0};

                for (int j = 0; j < n; j++)
                    src[j] = rand() % x;

                abcdk_enigma_light_batch(send_ctx, dst, src, n);
                //     printf("---------------------------\n");
                abcdk_enigma_light_batch(recv_ctx, dst2, dst, n);

                int chk = memcmp(src, dst2, n);
                assert(chk == 0);
            }

            abcdk_enigma_destroy(&send_ctx);
            abcdk_enigma_destroy(&recv_ctx);
        }
    }
#elif 0

    const char *src_file_p = abcdk_option_get(args,"--src-file",0,"aaaaa.mp4");
    const char *dst_file_p = abcdk_option_get(args,"--dst-file",0,"bbbbb.bin");
    const char *dst_file_p2 = abcdk_option_get(args,"--dst2-file",0,"bbbbb2.mp4");

    abcdk_enigma_t *s_ctx = abcdk_enigma_create2(2024,3,256);
    abcdk_enigma_t *r_ctx = abcdk_enigma_create2(2024,3,256);

    abcdk_object_t *src_data = abcdk_mmap_filename(src_file_p,0,0,0,0);
    abcdk_object_t *dst_data = abcdk_object_alloc2(src_data->sizes[0]);
    abcdk_object_t *dst_data2 = abcdk_object_alloc2(src_data->sizes[0]);

    for (int i = 0; i < 10; i++)
    {
        abcdk_enigma_light_batch(s_ctx, dst_data->pptrs[0], src_data->pptrs[0], src_data->sizes[0]);
        abcdk_enigma_light_batch(r_ctx, dst_data2->pptrs[0], dst_data->pptrs[0], src_data->sizes[0]);

        abcdk_save(dst_file_p,dst_data->pptrs[0],dst_data->sizes[0],0);
        abcdk_save(dst_file_p2,dst_data2->pptrs[0],dst_data2->sizes[0],0);
    }

    abcdk_object_unref(&src_data);
    abcdk_object_unref(&dst_data);
    abcdk_object_unref(&dst_data2);

    abcdk_enigma_free(&s_ctx);
    abcdk_enigma_free(&r_ctx);
#elif 0


    abcdk_enigma_t *s_ctx = abcdk_enigma_create2(2024,3,256);
    abcdk_enigma_t *r_ctx = abcdk_enigma_create2(2024,3,256);

    abcdk_object_t *src_data = abcdk_object_alloc2(1000);
    abcdk_object_t *dst_data = abcdk_object_alloc2(src_data->sizes[0]);
    abcdk_object_t *dst_data2 = abcdk_object_alloc2(src_data->sizes[0]);

    memset(src_data->pptrs[0],'a',src_data->sizes[0]);

    for (int i = 0; i < 10; i++)
    {
        abcdk_enigma_light_batch8(s_ctx, dst_data->pptrs[0], src_data->pptrs[0], src_data->sizes[0]);
        abcdk_enigma_light_batch8(r_ctx, dst_data2->pptrs[0], dst_data->pptrs[0], src_data->sizes[0]);

        abcdk_hexdump(stderr,dst_data->pptrs[0],dst_data->sizes[0],0,NULL);

    }

    abcdk_object_unref(&src_data);
    abcdk_object_unref(&dst_data);
    abcdk_object_unref(&dst_data2);

    abcdk_enigma_free(&s_ctx);
    abcdk_enigma_free(&r_ctx);
#elif 0

    abcdk_receiver_t *t = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_HTTP,100000,NULL);

    char data[] = {"POST /api HTTP/1.1\r\n"
    "Host: 127.0.0.1:17008\r\n"
    "User-Agent: python-requests/2.22.0\r\n"
    "Accept-Encoding: gzip, deflate\r\n"
    "Accept: */*\r\n"
    "Connection: keep-alive\r\n"
    "Content-Length: 489\r\n"
    "Content-Type: application/x-www-form-urlencoded\r\n"
    "\r\n"};

    size_t remain = 0;
    int chk = abcdk_receiver_append(t,data,strlen(data),&remain);

    abcdk_receiver_unref(&t);
#elif 0

    uint64_t a,b;

    abcdk_clock(0,&a);

    usleep(40*1000);

    b = abcdk_clock(a,&a);

    printf("b=%lu\n",b);
#elif 0

    uint64_t pos[2] = {1,0};

    for(int i = 0;i<10000;i++)
    {
        abcdk_save("/tmp/ccc/segment.log","aaaa",4,0);

        abcdk_file_segment("/tmp/ccc/segment.log","/tmp/ccc/segment.%llu.log",2,10000000,pos);
    }
#elif 0

    for (int i = 0; i < 10; i++)
    {
        char buf[16] = {0};
        abcdk_dmi_get_machine_hashcode(buf, 11, "test");

        abcdk_hexdump(stderr, buf, 16, 0, NULL);
    }

#elif 0

    //abcdk_option_merge(args,args);
        // 身份验证成功，进行其他操作...
    fprintf(stderr, "euid:%d,uid:%d\n", geteuid(), getuid());
#elif 0

    for(int i = 0;environ[i];i++)
    {
        fprintf(stderr,"%s\n",environ[i]);
    }

    char *param[] ={"--listen","0.0.0.0:1111","--root-path", "/home/data/files-b/", "--auto-index"};

    pid_t p = abcdk_exec_new("./abcdk",param,NULL,0,0,NULL,NULL,NULL,NULL,NULL);

    waitpid(p,NULL,0);

#elif 0

    const char *str = abcdk_option_get(args,"--str",0,"aaaa,bbbb,,,dddd,,ccc,,,,dd,,e,e,e,e");
    const char *delim = abcdk_option_get(args,"--delim",0,",");

    abcdk_object_t *buf = abcdk_strtok2vector(str,delim);

    for(int i = 0;i<buf->numbers;i++)
    {
        fprintf(stderr,"[%d]:(%zd)%s\n",i,buf->sizes[i],buf->pstrs[i]);
    }

    abcdk_object_unref(&buf);
#elif 0

#ifdef CURLINC_CURL_H

    const char *src = abcdk_option_get(args,"--src",0,"");
    const char *dst = abcdk_option_get(args,"--dst",0,"");
    const char *dst2 = abcdk_option_get(args,"--dst2",0,"");
    size_t offset = abcdk_option_get_int(args,"--offset",0,0);
    size_t count = abcdk_option_get_int(args,"--count",0,0);
    int chk = abcdk_curl_download_filename(dst,src,offset,count,6,6);
    assert(chk == 0);

    int fd = abcdk_open(dst2,1,0,1);

    chk = abcdk_curl_download_fd(fd,src,0,100,6,6);
    assert(chk == 0);

    chk = abcdk_curl_download_fd(fd,src,100,200,6,6);
    assert(chk == 0);

    chk = abcdk_curl_download_fd(fd,src,200,0,6,6);
    assert(chk == 0);

    abcdk_closep(&fd);

#endif //CURLINC_CURL_H

#elif 0

    abcdk_sockaddr_t dst = {0};
    int chk = abcdk_sockaddr_from_string(&dst, "ocsp.sectigochina.com", 1);
    assert(chk == 0);

#elif 0

    const char *p ="Digest username=\"aaaa\", realm=\"proxy\", nonce=\"147106898062910\", uri=\"/0584000065A09D5FC847B71286DAF47E?x-oss-process=image/resize,w_352/interlace,1/quality,Q_80\", response=\"37ebc4af75f6dcb98bd34e55f4583b02\"";

    abcdk_option_t *auth_opt =NULL;
    abcdk_http_parse_auth(&auth_opt,p);
    abcdk_option_free(&auth_opt);

#elif 0

    const char *src = "/home/devel/下载/aaaa.txt";
    const char *dst = "/home/devel/下载/aaaa.jpg";

    abcdk_object_t *buf = abcdk_mmap_filename(src,0,0,0,0);

    abcdk_object_t *buf2 = abcdk_basecode_decode2(buf->pptrs[0],buf->sizes[0],64);

    abcdk_save((char *)dst,buf2->pptrs[0],buf2->sizes[0],0);


    abcdk_object_unref(&buf2);
    abcdk_object_unref(&buf);
    
#elif 0

    abcdk_object_t *f = abcdk_mmap_filename("/home/devel/job/tmp/c.bmp",0,0,0,0);

    abcdk_package_t *ctx = abcdk_package_create(100000000);

    abcdk_package_write_number(ctx,16,333);
    abcdk_package_write_number(ctx,8,33);

    abcdk_package_write_buffer(ctx,f->pptrs[0],f->sizes[0]);
    
    abcdk_package_write_number(ctx,24,555);

    abcdk_object_t * obj = abcdk_package_dump(ctx,1);

    abcdk_package_destroy(&ctx);

    ctx = abcdk_package_load(obj->pptrs[0],obj->sizes[0]);
    abcdk_object_unref(&obj);

    uint64_t a = abcdk_package_read2number(ctx,16);
    uint64_t b = abcdk_package_read2number(ctx,8);

    abcdk_package_seek(ctx,f->sizes[0]*8);

    uint64_t c = abcdk_package_read2number(ctx,24);


    abcdk_package_destroy(&ctx);

    abcdk_object_unref(&f);

#elif 0
    abcdk_object_t *src_data = abcdk_mmap_filename("/home/devel/job/tmp/c.bmp",0,0,0,0);
    abcdk_object_t *src_data2 = abcdk_object_alloc2(src_data->sizes[0]);
    
    abcdk_object_t *dst_data = abcdk_object_alloc2(src_data->sizes[0]);

    uint64_t s;
    abcdk_clock(s, &s);
    for (int i = 0; i < 1000; i++)
    {
        abcdk_trace_output(LOG_INFO, "enc-b:%06llu", abcdk_clock(s, &s));
        int m = abcdk_lz4_enc(dst_data->pptrs[0], dst_data->sizes[0], src_data->pptrs[0], src_data->sizes[0]);
        abcdk_trace_output(LOG_INFO, "enc-a:%06llu", abcdk_clock(s, &s));

        abcdk_trace_output(LOG_INFO, "dec-b:%06llu", abcdk_clock(s, &s));
        int n = abcdk_lz4_dec(src_data2->pptrs[0], src_data2->sizes[0], dst_data->pptrs[0],m);
        abcdk_trace_output(LOG_INFO, "dec-a:%06llu", abcdk_clock(s, &s));
        
        assert(n = src_data->sizes[0]);

        usleep(1000 * 100);
    }

    abcdk_object_unref(&src_data);
    abcdk_object_unref(&src_data2);
    abcdk_object_unref(&dst_data);

#elif 0

    abcdk_package_t *src = abcdk_package_create(100);

    abcdk_package_write_number(src,4,1);
    abcdk_package_write_number(src,12,401);
    abcdk_package_write_number(src,16,23456);
    abcdk_package_write_number(src,13,456);
    abcdk_package_write_number(src,3,2);
    abcdk_package_write_number(src,64,99999999999999LL);
    abcdk_package_write_number(src,32,999999999);

    abcdk_object_t *buf = abcdk_package_dump(src,0);
    abcdk_package_destroy(&src);

    abcdk_package_t *dst = abcdk_package_load(buf->pptrs[0],buf->sizes[0]);


    int64_t a = abcdk_package_read2number(dst,4);
    int64_t b = abcdk_package_read2number(dst,12);
    int64_t c = abcdk_package_read2number(dst,16);
    int64_t d = abcdk_package_read2number(dst,13);
    int64_t f = abcdk_package_read2number(dst,3);
    int64_t g = abcdk_package_read2number(dst,64);
    int64_t h = abcdk_package_read2number(dst,32);

    abcdk_package_destroy(&dst);
#elif 0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                       unsigned int buf[20] = {0};

        int c = abcdk_get_cpuid(buf);

        for(int i=0;i<c;i++)
        {
            fprintf(stderr,"%08x ",buf[i]);
        }

        fprintf(stderr,"\n");
#elif 0

#ifdef __NCURSES_H

    // 初始化 ncurses
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);

    // 获取屏幕的大小
    int max_y, max_x;
    getmaxyx(stdscr, max_y, max_x);

    // 创建水平分割窗口
    // WINDOW *a = newwin(max_y / 2, max_x, 0, 0);
    // WINDOW *b = newwin(max_y / 2, max_x, max_y / 2, 0);
    
    //创建垂直分割窗口
    WINDOW *a = newwin(max_y, max_x / 2, 0, 0);
    WINDOW *b = newwin(max_y, max_x / 2, 0, max_x / 2);

    // 在每个窗口中创建菜单
    wprintw(a, "Upper Window Menu:\n");
    wprintw(a, "1. Option 1\n");
    wprintw(a, "2. Option 2\n");
    wprintw(a, "3. Option 3\n");
    wprintw(a, "4. Exit\n");

    wprintw(b, "Lower Window Menu:\n");
    wprintw(b, "a. Suboption 1\n");
    wprintw(b, "b. Suboption 2\n");
    wprintw(b, "c. Back\n");

    // 刷新屏幕
    refresh();
    wrefresh(a);
    wrefresh(b);

    // 等待用户按下任意键后退出
    getch();

    // 清理 ncurses 并退出程序
    delwin(a);
    delwin(b);
    endwin();

#endif //__NCURSES_H
#elif 0

    for(int i = 0;i<5;i++)
    {
        char buf[123] = {0};
        abcdk_rand_bytes(buf,122,i);

        fprintf(stderr,"[_%s_]\n",buf);
    }
#elif 0

    char buf[100] = {0};

    abcdk_sha256_from_buffer2string("",0,buf,0);

    printf("%s\n",buf);

    char buf2[100] = {0};

    abcdk_sha256_from_buffer2string("abc",3,buf2,0);

    printf("%s\n",buf2);

#elif 0

    //abcdk_thread_setaffinity2(pthread_self(),4);

    for (int r = 3; r <= 10; r++)
    {
        abcdk_enigma_t *s_ctx = abcdk_enigma_create(r,256);
        abcdk_enigma_t *r_ctx = abcdk_enigma_create(r,256);

        abcdk_enigma_init_ex(s_ctx,"aaa",3);
        abcdk_enigma_init_ex(r_ctx,"aaa",3);

        for (int d = 1000; d <= 10*1000 * 1000; d *= 10)
        {

#if 0
            uint64_t s = 0;
            abcdk_clock(s,&s);

            for(int i = 0;i<d;i++)
            {
                uint8_t d1 = abcdk_enigma_light(s_ctx,'a');
                uint8_t d2 = abcdk_enigma_light(r_ctx,d1);
                assert(d2 == 'a');
            }
            
            uint64_t step = abcdk_clock(s,&s);
            fprintf(stderr,"r(%d),d(%d),step(%0.9f)\n",r,d,(double)step/1000000000.);
            
#else 

            abcdk_object_t *src_data = abcdk_object_alloc2(d);
            abcdk_object_t *dst_data = abcdk_object_alloc2(d);
            abcdk_object_t *dst_data2 = abcdk_object_alloc2(d);
#ifndef HAVE_OPENSSL
            abcdk_rand_bytes(src_data->pptrs[0], d, 0);
#else 
            RAND_bytes(src_data->pptrs[0], d);
#endif 

            uint64_t s = 0;
            abcdk_clock(s,&s);

                     
            abcdk_enigma_light_batch(s_ctx, dst_data->pptrs[0], src_data->pptrs[0], d);
            abcdk_enigma_light_batch(r_ctx, dst_data2->pptrs[0], dst_data->pptrs[0], d);
            assert(memcmp(src_data->pptrs[0],dst_data2->pptrs[0],d)==0);

            uint64_t step = abcdk_clock(s,&s);
            fprintf(stderr,"r(%d),d(%d),step(%0.9f)\n",r,d,(double)step/1000000000.);
            

            abcdk_object_unref(&src_data);
            abcdk_object_unref(&dst_data);
            abcdk_object_unref(&dst_data2);
#endif
        }

        abcdk_enigma_destroy(&s_ctx);
        abcdk_enigma_destroy(&r_ctx);
    }
#elif 0

    int pipefd[2] = {-1,-1};

    pipe(pipefd);

    abcdk_fflag_set(pipefd[0],O_NONBLOCK);
    abcdk_fflag_set(pipefd[1],O_NONBLOCK);

    abcdk_easyssl_t *cli_ctx = abcdk_easyssl_create("abc123",6,1,33);


    abcdk_easyssl_set_fd(cli_ctx,pipefd[0],0);
    abcdk_easyssl_set_fd(cli_ctx,pipefd[1],1);

#pragma omp parallel for num_threads(2)
    for(int i = 0;i<2;i++)
    {
        if(i == 0)
        {
            size_t buf_l = 1920*1080*3;
            char *buf_p = (char*)abcdk_heap_alloc(buf_l);
            for(int k = 0;k<1080*3;k++)
                memset(buf_p+k*1920,ABCDK_CLAMP(k%256,65,90),1920);

            for(int j = 0;j<10;j++)
            {
                size_t sn = 0;
                while (sn < buf_l)
                {
                    int n = abcdk_easyssl_write(cli_ctx, buf_p + sn, buf_l - sn);
                    if (n < 0)
                        abcdk_poll(pipefd[1], 0x02, -1);
                    else if (n == 0)
                        break;
                    else
                        sn += n;
                }

                if(sn <=0)
                    break;

                printf("sn(%zd)\n",sn);

            }

            sleep(2);
            

            abcdk_closep(&pipefd[1]);
        }
        else 
        {
            
            for (int j = 0; j < 10; j++)
            {
                for (int k = 0; k < 1080 * 3; k++)
                {
                    size_t rn = 0;
                    char buf[1920 + 1] = {0};
                    while (rn < 1920)
                    {

                        int n = abcdk_easyssl_read(cli_ctx, buf + rn, 1920 - rn);
                        if (n < 0)
                            abcdk_poll(pipefd[0], 0x01, -1);
                        else if (n == 0)
                            break;
                        else
                            rn += n;
                    }

                    printf("%s\n", buf);
                }
            }

            abcdk_closep(&pipefd[0]);
        }

    }

    


    abcdk_easyssl_destroy(&cli_ctx);

#elif 0

    abcdk_ffserver_config_t src_cfg ={0};
    abcdk_ffserver_config_t push_cfg ={0};
    abcdk_ffserver_config_t record_cfg ={0};

   // src_cfg.u.src.url = "/home/devel/job/download/4K PARADISE Summer Mix 2024 🍓 Best Of Tropical Deep House Music Chill Out Mix By Summer Vibes Sound.mp4";
    src_cfg.u.src.url = "rtsp://192.168.100.96/live/bbbb";
    src_cfg.u.src.speed = 1.0;
    src_cfg.u.src.delay_max = 1.0;

    push_cfg.flag = 2;
    push_cfg.u.push.url = "rtmp://192.168.100.96/live/cccc";
    push_cfg.u.push.fmt = "rtmp";

    record_cfg.flag = 1;
    record_cfg.u.record.prefix = "/home/devel/job/tmp/cccc/cccc_";
    record_cfg.u.record.count = 10;
    record_cfg.u.record.duration = 5;

    abcdk_ffserver_t *ctx = abcdk_ffserver_create(&src_cfg);

    abcdk_ffserver_task_add(ctx,&push_cfg);
    abcdk_ffserver_task_add(ctx,&record_cfg);

    while(getchar() != 'q');

    abcdk_ffserver_destroy(&ctx);

#elif 0

    abcdk_sockaddr_t dst = {0};
    char buf[100] = {0};
    int chk = abcdk_sockaddr_from_string(&dst, abcdk_option_get(args, "--ipaddr", 0, ""), 1);
    assert(chk == 0);

    abcdk_sockaddr_to_string(buf, &dst,0);
    abcdk_trace_output(LOG_DEBUG, "%s\n", buf);

#elif 0

    
    uint32_t ip_start = 10;
    ip_start <<= 8;
    ip_start += 0;
    ip_start <<= 8;
    ip_start += 0;
    ip_start <<= 8;
    ip_start += 0;
    

    int ip_prefix = 23;
    uint32_t ip_count = 0;
    for(int i = 0;i<32-ip_prefix;i++)
    {
        ip_count <<=1;
        ip_count +=1;
    }

    for (int i = 0; i < ip_count; i++)
    {

        abcdk_sockaddr_t dst = {AF_INET};
        dst.addr4.sin_addr.s_addr = abcdk_endian_h_to_b32(ip_start + i);

        char buf[100] = {0};
        abcdk_sockaddr_to_string(buf, &dst,0);
        abcdk_trace_output(LOG_DEBUG, "%s\n", buf);
    }

#elif 0


    abcdk_ipool_t *ctx = abcdk_ipool_create2(abcdk_option_get(args, "--start", 0, ""),abcdk_option_get(args, "--end", 0, ""));

    int chk = abcdk_ipool_set_dhcp_range2(ctx,abcdk_option_get(args, "--dhcp-start", 0, ""),abcdk_option_get(args, "--dhcp-end", 0, ""));

    chk = abcdk_ipool_static_request2(ctx,abcdk_option_get(args, "--static", 0, ""));
    assert(chk == 0);

    int c = abcdk_ipool_count(ctx,2);
    int i = 0;
    for(;i<c;i++)
    {
        abcdk_sockaddr_t addr = {0};
        chk = abcdk_ipool_dhcp_request(ctx,&addr);
        if(chk != 0)
            break;
        
        char buf[100] = {0};
        abcdk_sockaddr_to_string(buf, &addr);
        abcdk_trace_output(LOG_DEBUG, "%s\n", buf);

        abcdk_ipool_reclaim(ctx,&addr);
        
    }

    
    int j = 0;
    for(;j<c;j++)
    {
        abcdk_sockaddr_t addr = {0};
        chk = abcdk_ipool_dhcp_request(ctx,&addr);
        if(chk != 0)
            break;

        abcdk_ipool_reclaim(ctx,&addr);
    }

    assert(i == j);

    abcdk_ipool_destroy(&ctx);

#elif 1

    abcdk_ipool_t *pool_ctx = abcdk_ipool_create2("192.168.123.1","192.168.123.254");

    int chk = abcdk_ipool_set_dhcp_range2(pool_ctx,"192.168.123.1","192.168.123.254");

    abcdk_iplan_t *plan_ctx = abcdk_iplan_create(0);

    int c = abcdk_ipool_count(pool_ctx,2);
    for(int i = 0;i<c;i++)
    {
        abcdk_sockaddr_t addr = {0};
        chk = abcdk_ipool_dhcp_request(pool_ctx,&addr);
        if(chk != 0)
            break;

        abcdk_iplan_insert(plan_ctx,&addr,(void*)(long)(i+1));
 
        abcdk_ipool_reclaim(pool_ctx,&addr);
    }

    abcdk_iplan_iterator_t *plan_it = NULL;
    void *addr_p = NULL;

    abcdk_sockaddr_t addr = {0};

    abcdk_sockaddr_from_string(&addr,"192.168.123.123",0);

    abcdk_iplan_remove(plan_ctx,&addr);

    while(addr_p = abcdk_iplan_next(plan_ctx,&plan_it))
    {
        fprintf(stderr,"%p\n",addr_p);
    }

    abcdk_iplan_destroy(&plan_ctx);
    abcdk_ipool_destroy(&pool_ctx);
#elif 0

#ifdef HAVE_OPENSSL

   //abcdk_openssl_cipher_t *enc_ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_GCM,"aaaa",4);
   //abcdk_openssl_cipher_t *dec_ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_GCM,"aaaa",4);

      abcdk_openssl_cipher_t *enc_ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_CBC,"aaaa",4);
    abcdk_openssl_cipher_t *dec_ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_CBC,"aaaa",4);

    int bufsize = 2000;
    char buf[bufsize];
    memset(buf,'a',bufsize);

    abcdk_object_t *buf2 = abcdk_openssl_cipher_update(enc_ctx,buf,bufsize,1);
    abcdk_object_t *buf3 = abcdk_openssl_cipher_update(enc_ctx,buf,bufsize,1);

    abcdk_object_t *buf4 = abcdk_openssl_cipher_update(dec_ctx,buf2->pptrs[0],buf2->sizes[0],0);
    abcdk_object_t *buf5 = abcdk_openssl_cipher_update(dec_ctx,buf3->pptrs[0],buf3->sizes[0],0);

    assert(memcmp(buf4->pptrs[0],buf5->pptrs[0],bufsize)==0);
    assert(memcmp(buf,buf5->pptrs[0],bufsize)==0);

    abcdk_object_unref(&buf2);
    abcdk_object_unref(&buf3);
    abcdk_object_unref(&buf4);
    abcdk_object_unref(&buf5);

    abcdk_object_t *src = abcdk_object_alloc2(10000000);

        
    uint64_t dot = 0;
    abcdk_clock(dot,&dot);

    for (int i = 1000; i <= 10000000; i *= 10)
    {

        RAND_bytes(src->pptrs[0], i);

        abcdk_clock(dot,&dot);

#if 0
        abcdk_object_t *buf6 = abcdk_openssl_cipher_update(enc_ctx, src->pptrs[0], i, 1);

        abcdk_object_t *buf7 = abcdk_openssl_cipher_update(dec_ctx, buf6->pptrs[0], buf6->sizes[0], 0);

        assert(memcmp(src->pptrs[0], buf7->pptrs[0], i) == 0);
#else 
        abcdk_object_t *buf6 = abcdk_openssl_cipher_update_pack(enc_ctx, src->pptrs[0], i, 1);

        abcdk_object_t *buf7 = abcdk_openssl_cipher_update_pack(dec_ctx, buf6->pptrs[0], buf6->sizes[0], 0);

        assert(buf7->sizes[0] == i);
        assert(memcmp(src->pptrs[0], buf7->pptrs[0], i) == 0);
#endif 

        abcdk_object_unref(&buf6);
        abcdk_object_unref(&buf7);

        uint64_t step = abcdk_clock(dot, &dot);
        fprintf(stderr, "%d,cast:%.9f\n", i, (double)step / 1000000000.);
    }

    abcdk_object_unref(&src);

    abcdk_openssl_cipher_destroy(&enc_ctx);
    abcdk_openssl_cipher_destroy(&dec_ctx);

#endif //HAVE_OPENSSL

#elif 0


#ifdef HAVE_OPENSSL

    const char *prikey_file = abcdk_option_get(args, "--private-key-file", 0, "");
    const char *pubkey_file = abcdk_option_get(args, "--public-key-file", 0, "");

    abcdk_openssl_cipher_t *pri_cipher = abcdk_openssl_cipher_create_from_file(ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PRIVATE,prikey_file);
    abcdk_openssl_cipher_t *pub_cipher = abcdk_openssl_cipher_create_from_file(ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PUBLIC,pubkey_file);

    int bufsize = 2000;
    char buf[bufsize];
    memset(buf,'a',bufsize);


    abcdk_object_t *buf2 = abcdk_openssl_cipher_update(pri_cipher,buf,bufsize,1);
    abcdk_object_t *buf3 = abcdk_openssl_cipher_update(pri_cipher,buf,bufsize,1);

    abcdk_object_t *buf4 = abcdk_openssl_cipher_update(pub_cipher,buf2->pptrs[0],buf2->sizes[0],0);
    abcdk_object_t *buf5 = abcdk_openssl_cipher_update(pub_cipher,buf3->pptrs[0],buf3->sizes[0],0);

    assert(memcmp(buf4->pptrs[0],buf5->pptrs[0],bufsize)==0);
    assert(memcmp(buf,buf5->pptrs[0],bufsize)==0);

    abcdk_object_unref(&buf2);
    abcdk_object_unref(&buf3);
    abcdk_object_unref(&buf4);
    abcdk_object_unref(&buf5);

    abcdk_object_t *src = abcdk_object_alloc2(100000);

    uint64_t dot = 0;
    abcdk_clock(dot,&dot);

    for(int i = 1;i<10000;i++)
    {
        RAND_bytes(src->pptrs[0],i);

        abcdk_object_t *buf6 = abcdk_openssl_cipher_update(i%2?pri_cipher:pub_cipher,src->pptrs[0],i,1);

        abcdk_object_t *buf7 = abcdk_openssl_cipher_update(i%2?pub_cipher:pri_cipher,buf6->pptrs[0],buf6->sizes[0],0);

        assert(memcmp(src->pptrs[0],buf7->pptrs[0],i)==0);

        abcdk_object_unref(&buf6);
        abcdk_object_unref(&buf7);
    }

    uint64_t step = abcdk_clock(dot,&dot);
    fprintf(stderr,"cast:%.6f\n",(double)step/1000000.);

    abcdk_object_unref(&src);

    abcdk_openssl_cipher_destroy(&pri_cipher);
    abcdk_openssl_cipher_destroy(&pub_cipher);

#endif //HAVE_OPENSSL


#elif 0

#ifdef HAVE_OPENSSL

    const char *ca_file = abcdk_option_get(args, "--ca-file", 0, NULL);
    const char *ca_path = abcdk_option_get(args, "--ca-path", 0, NULL);
    const char *cert_file = abcdk_option_get(args, "--cert-file", 0, NULL);

    X509_STORE *store = abcdk_openssl_cert_load_locations(ca_file,ca_path);

    int chk = X509_STORE_set_flags(store, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);
    assert(chk == 1);

    X509 *leaf_cert = abcdk_openssl_cert_load(cert_file,NULL);
    assert(leaf_cert != NULL);

    STACK_OF(X509) *cert_chain = abcdk_openssl_cert_chain_load(leaf_cert,ca_path,"*.crt");
    assert(cert_chain != NULL);

    abcdk_object_t *cert_chain_pem = abcdk_openssl_cert_to_pem(leaf_cert,cert_chain);
    assert(cert_chain_pem);
    fprintf(stderr,"%s\n",cert_chain_pem->pstrs[0]);
    abcdk_object_unref(&cert_chain_pem);

    X509_STORE_CTX *store_ctx = abcdk_openssl_cert_verify_prepare(store, leaf_cert,cert_chain);
    assert(store_ctx != NULL);
    
    chk = X509_verify_cert(store_ctx);
    if(chk != 1)
    {
        abcdk_object_t *err_info = abcdk_openssl_cert_verify_error_dump(store_ctx);
        assert(err_info);
        fprintf(stderr,err_info->pstrs[0]);
        abcdk_object_unref(&err_info);
    }

    X509_free(leaf_cert);
    sk_X509_pop_free(cert_chain, X509_free);
    X509_STORE_CTX_free(store_ctx);
    X509_STORE_free(store);

#endif //HAVE_OPENSSL

#elif 0

    abcdk_object_t *tmp = abcdk_object_alloc2(100000);

    uint64_t dot = 0;
    abcdk_clock(dot, &dot);

    for (int i = 1; i < 10000; i++)
    {
        //RAND_bytes(tmp->pptrs[0], 65000);
        abcdk_rand_bytes(tmp->pptrs[0], 65000,4);

        abcdk_package_t *src = abcdk_package_create(65535);

        abcdk_package_write_buffer(src, tmp->pptrs[0], 65000);

        abcdk_object_t *data = abcdk_package_dump(src,1);

        abcdk_object_unref(&data);

        abcdk_package_destroy(&src);
    }

    uint64_t step = abcdk_clock(dot, &dot);
    fprintf(stderr, "cast:%.6f\n", (double)step / 1000000.);

    abcdk_object_unref(&tmp);
#elif 0

    // srand(time(0)); // 初始化随机数种子

     abcdk_wred_t *ctx = abcdk_wred_create(200,400,2,2);

    int drop_count = 0;    

    for (int i = 0; i < 10000; i++)
    {
        int qlen = rand() % 500+10;
        int drop = abcdk_wred_update(ctx, qlen);

        drop_count += (drop?1:0);

        printf("idx(%d),qlen(%d): %s\n",i+1, qlen, drop ? "丢弃" : "保留");
    }

    printf("丢包数量：%d\n",drop_count);

    abcdk_wred_destroy(&ctx);

#elif 0

            uint64_t s = 0;
            abcdk_clock(s,&s);

            uint8_t buf[256];
            int c;

            for(int i = 0;i<10*1000*1000;i+=3)
            {
                c = buf[i%256];
            }
            
            uint64_t step = abcdk_clock(s,&s);
            fprintf(stderr,"cast: %0.9f\n",(double)step/1000000000.);

#elif 0

#ifdef HAVE_OPENSSL

    EVP_CIPHER_CTX *enc_evp_ctx = EVP_CIPHER_CTX_new();
    EVP_CIPHER_CTX *dec_evp_ctx = EVP_CIPHER_CTX_new();

    char key[32] = {0},iv[16] = {0};

    RAND_bytes(key,32);
    RAND_bytes(iv,16);

    EVP_CipherInit_ex(enc_evp_ctx, EVP_aes_256_ctr(), NULL, NULL, NULL, 1);
    EVP_CipherInit_ex(dec_evp_ctx, EVP_aes_256_ctr(), NULL, NULL, NULL, 0);

    assert(EVP_CIPHER_CTX_key_length(enc_evp_ctx) == 32);
    assert(EVP_CIPHER_CTX_iv_length(enc_evp_ctx) == 16);

    assert(EVP_CipherInit_ex(enc_evp_ctx, NULL, NULL, key, iv, 1)==1);
    assert(EVP_CipherInit_ex(dec_evp_ctx, NULL, NULL, key, iv, 0)==1);

    uint8_t enc_inbuf[1000];
    uint8_t enc_outbuf[1000];
    uint8_t dec_outbuf[1000];

    memset(enc_inbuf,'a',1000);

#if 0
    int enc_outlen;
    assert(EVP_CipherUpdate(enc_evp_ctx, enc_outbuf, &enc_outlen, enc_inbuf, 1000)==1);
#else 
    for(int i = 0;i<1000;i++)
    {   
        int enc_outlen;
        assert(EVP_CipherUpdate(enc_evp_ctx, enc_outbuf+1, &enc_outlen, enc_inbuf+1, 1)==1);
    }
#endif 

#if 1 
    int dec_outlen;
    assert(EVP_CipherUpdate(dec_evp_ctx, dec_outbuf, &dec_outlen, enc_outbuf, 1000)==1);
#else 
    for(int i = 0;i<1000;i++)
    {
        int dec_outlen;
        assert(EVP_CipherUpdate(dec_evp_ctx, dec_outbuf+i, &dec_outlen, enc_outbuf+i, 1)==1);
    }
#endif 

    memcmp(enc_inbuf,dec_outbuf,1000);

    EVP_CIPHER_CTX_free(enc_evp_ctx);
    EVP_CIPHER_CTX_free(dec_evp_ctx);

#endif //HAVE_OPENSSL


#endif 
    return 0;
}
