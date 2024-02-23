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
    
#elif 1

    abcdk_object_t *f = abcdk_mmap_filename("/home/devel/job/tmp/c.bmp",0,0,0,0);

    abcdk_package_t *ctx = abcdk_package_create(100000000);

    abcdk_package_write_number(ctx,16,333);
    abcdk_package_write_number(ctx,8,33);

    abcdk_package_write_buffer(ctx,f->pptrs[0],f->sizes[0]);
    
    abcdk_package_write_number(ctx,24,555);

    abcdk_object_t * obj = abcdk_package_dump(ctx,1);

    abcdk_package_destroy(&ctx);

    ctx = abcdk_package_load(obj);
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
#endif 
}