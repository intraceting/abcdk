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
#include <wchar.h>
#include <sys/mount.h>
#include <sys/sendfile.h>
#include <linux/fb.h>
#include <linux/serial.h>
#include "util/general.h"
#include "util/getargs.h"
#include "util/geometry.h"
#include "util/ffmpeg.h"
#include "util/bmp.h"
#include "util/freeimage.h"
#include "util/uri.h"
#include "util/html.h"
#include "util/clock.h"
#include "util/crc32.h"
#include "util/robots.h"
#include "util/dirent.h"
#include "util/socket.h"
#include "util/hexdump.h"
#include "util/termios.h"
#include "mp4/demuxer.h"
#include "util/video.h"
#include "util/lz4.h"
#include "util/openssl.h"
#include "util/redis.h"
#include "comm/comm.h"
#include "comm/message.h"
#include "comm/queue.h"
#include "comm/waiter.h"
#include "util/json.h"
#include "comm/easy.h"
#include "util/base64.h"
#include "util/basecode.h"
#include "util/notify.h"
#include "util/scsi.h"
#include "util/ndarray.h"
#include "shell/mtab.h"
#include "shell/block.h"
#include "shell/mmc.h"
#include "shell/scsi.h"
#include "shell/file.h"
#include "util/iconv.h"
#include "util/sqlite.h"
#include "util/reader.h"
#include "util/json.h"
#include "util/signal.h"
#include "util/odbcpool.h"
#include "log/log.h"


#ifdef HAVE_FUSE
#define FUSE_USE_VERSION 29
#include <fuse.h>
#endif //

#ifdef HAVE_LIBNM
#include <NetworkManager.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif 


#ifdef HAVE_ARCHIVE
#include <archive.h>
#include <archive_entry.h>
#endif

#ifdef HAVE_MODBUS
#include <modbus.h>
#endif 

#ifdef HAVE_LIBUSB
#include <libusb.h>
#endif

#ifdef HAVE_MQTT
#include <mosquitto.h>
#endif 

#ifdef HAVE_BLKID
#include <blkid/blkid.h>
#endif

#ifdef HAVE_FASTCGI
#include <fcgiapp.h>
#endif

#ifdef HAVE_LIBUDEV
#include <libudev.h>
#endif

#ifdef HAVE_LIBDMTX
#include <dmtx.h>
#endif

#ifdef HAVE_QRENCODE
#include <qrencode.h>
#endif

#ifdef HAVE_ZBAR
#include <zbar.h>
#endif 

#ifdef HAVE_MAGICKWAND
#include <wand/MagickWand.h>
#endif

#ifdef HAVE_KAFKA
#include <librdkafka/rdkafka.h>
#endif 

void* sigwaitinfo_cb(void* args)
{
    abcdk_signal_t sig;
    sigfillset(&sig.signals);

    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    sig.signal_cb = NULL;
    sig.opaque = NULL;
     
    abcdk_sigwaitinfo(&sig,-1);

    return NULL;
}

void* server_loop(void* args)
{
    abcdk_epollex_t *m = (abcdk_epollex_t *)args;
    static volatile pthread_t leader = 0;
    static int l = -1;

    if (abcdk_thread_leader_vote(&leader) == 0)
    {

        l = abcdk_socket(AF_INET, 0);
        abcdk_sockaddr_t a = {0};
        abcdk_sockaddr_from_string(&a, "localhost:12345", 1);

        int flag = 1;
        abcdk_sockopt_option_int(l, SOL_SOCKET, SO_REUSEPORT, &flag, 2);
        abcdk_sockopt_option_int(l, SOL_SOCKET, SO_REUSEADDR, &flag, 2);

        abcdk_bind(l, &a);
        listen(l, SOMAXCONN);

        assert(abcdk_epollex_attach2(m, l) == 0);
        assert(abcdk_epollex_mark(m, l, ABCDK_EPOLL_INPUT, 0) == 0);
    }

    while(1)
    {
        abcdk_epoll_event_t e;
        int chk = abcdk_epollex_wait(m, &e, -1);
        if (chk < 0)
            break;

        if(e.events & ABCDK_EPOLL_ERROR)
        {
            //assert(abcdk_epollex_mark(m,e.data.fd,0,e.events)==0);
            assert(abcdk_epollex_unref(m,e.data.fd,e.events)==0);
            assert(abcdk_epollex_detach(m,e.data.fd)==0);
            abcdk_closep(&e.data.fd);
        }
        else if(e.data.fd == l)
        {
            while(1)
            {
                int c = abcdk_accept(e.data.fd,NULL);
                if(c<0)
                    break;

                int flag=1;
                assert(abcdk_sockopt_option_int(c, IPPROTO_TCP, TCP_NODELAY,&flag, 2) == 0);

                assert(abcdk_epollex_attach2(m, c) == 0);
                assert(abcdk_epollex_timeout(m, c, 5*1000) == 0);
                assert(abcdk_epollex_mark(m,c,ABCDK_EPOLL_INPUT,0)==0);
            }
            
            assert(abcdk_epollex_mark(m,e.data.fd,ABCDK_EPOLL_INPUT,ABCDK_EPOLL_INPUT)==0);
        }
        else
        {
            if(e.events & ABCDK_EPOLL_INPUT)
            {
                while (1)
                {
                    char buf[100];
                    int r = recv(e.data.fd, buf, 100, 0);
                    if (r > 0)
                    {
                        printf("%s\n",buf);
                        continue;
                    }
                    if (r == -1 && errno == EAGAIN)
                        assert(abcdk_epollex_mark(m,e.data.fd, ABCDK_EPOLL_INPUT|ABCDK_EPOLL_OUTPUT, ABCDK_EPOLL_INPUT) == 0);

                    break;
                }
            }
            if(e.events & ABCDK_EPOLL_OUTPUT)
            {
                while (1)
                {
                    static int fd = -1;
                    if(fd ==-1)
                        fd = abcdk_open("/var/log/kdump.log",0,0,0);
#if 0 
                    char buf[100];
                    size_t n = abcdk_read(fd, buf, 100);
                    if (n <= 0)
                    {
                        abcdk_closep(&fd);
                        assert(abcdk_epollex_mark(m,e.data.fd, 0, ABCDK_EPOLL_OUTPUT) == 0);
                        break;
                    }

                    //memset(buf,rand()%26+40,100);
                    int s = send(e.data.fd, buf, 100, 0);
                    if (s > 0)
                        continue;
                    if (s == -1 && errno == EAGAIN)
                        assert(abcdk_epollex_mark(m,e.data.fd, ABCDK_EPOLL_OUTPUT, ABCDK_EPOLL_OUTPUT) == 0);
#else
                    ssize_t s = sendfile(e.data.fd,fd,NULL,100000000);
                    printf("s=%ld\n",s);
                    if(s>0)
                        continue;
                    if (s == 0)
                    {
                        abcdk_closep(&fd);
                        assert(abcdk_epollex_mark(m,e.data.fd, 0, ABCDK_EPOLL_OUTPUT) == 0);
                        break;
                    }
                    else if (s == -1 && errno == EAGAIN)
                    {
                        assert(abcdk_epollex_mark(m,e.data.fd, ABCDK_EPOLL_OUTPUT, ABCDK_EPOLL_OUTPUT) == 0);
                    }
#endif
                    break;
                }
            }

            assert(abcdk_epollex_unref(m,e.data.fd,e.events)==0);
        }
        
    }

    return NULL;
}

void test_server(abcdk_tree_t *t)
{
 //   abcdk_clock_reset();

    abcdk_epollex_t *m = abcdk_epollex_alloc(NULL,NULL);
#if 0

    printf("attach begin:%lu\n",abcdk_clock_dot(NULL));

    for (int i = 0; i < 100000; i++)
        assert(abcdk_epollex_attach(m, i, 100) == 0);
    //   assert(abcdk_epollex_attach(m,10000,100)==0);

    printf("attach cast:%lu\n",abcdk_clock_step(NULL));

    getchar();
    
    printf("attach begin:%lu\n",abcdk_clock_dot(NULL));

    for (int i = 0; i < 100000; i++)
        assert(abcdk_epollex_detach(m, i) == 0);

    printf("detach cast:%lu\n",abcdk_clock_step(NULL));
#else

    #pragma omp parallel for num_threads(3)
    for (int i = 0; i < 3; i++)
    {
#if 0
        abcdk_thread_t p;
        p.routine = server_loop;
        p.opaque = m;
        abcdk_thread_create(&p, 0);
#else 
        server_loop(m);
#endif 
    }

    while (getchar()!='Q');    

#endif

    abcdk_epollex_free(&m);
}

void test_client(abcdk_tree_t *t)
{
    int c = abcdk_socket(AF_INET, 0);
    abcdk_sockaddr_t a = {0};
    abcdk_sockaddr_from_string(&a, abcdk_option_get(t,"--",1,"localhost:12345"), 1);

    int chk = abcdk_connect(c,&a,10000);
    assert(chk==0);

    send(c,"aaa\n",3,0);

    while(1)
    {   
        char buf[100] ={0};
        int r = recv(c,buf,100,0);
        if(r<=0)
            break;
        printf("%s\n",buf);
    }

    abcdk_closep(&c);
}

void test_mux(abcdk_tree_t *args)
{
    if(abcdk_option_exist(args,"--server"))
        test_server(args);
    else 
        test_client(args);
}

void test_ffmpeg(abcdk_tree_t *args)
{

#ifdef HAVE_FFMPEG

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,20,100)
    av_register_all();
#endif

    avdevice_register_all();

    for(int i = 0;i<1000;i++)
    {
        enum AVPixelFormat pixfmt = (enum AVPixelFormat)i;

        int bits = abcdk_av_image_pixfmt_bits(pixfmt,0);
        int bits_pad = abcdk_av_image_pixfmt_bits(pixfmt,1);
        const char *name = abcdk_av_image_pixfmt_name(pixfmt);
        if(!name)
            continue;

        printf("%s(%d): %d/%d bits.\n",name,i,bits,bits_pad);
    }

    AVInputFormat *in = NULL;
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,20,100)
    while(in = av_iformat_next(in))
#else 
    while(0)
#endif 
    {
        if(!in->priv_class)
            continue;
        
    //    printf("%s _ %s _ %s\n",in->name,in->long_name,in->mime_type);
        av_opt_show2((void *)&in->priv_class, NULL, -1, 0);
    }


    AVOutputFormat *out = NULL;
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58,20,100)
    while(out = av_oformat_next(out))
#else 
    while(0)
#endif 
    {
        if(!out->priv_class)
            continue;
         printf("%s _ %s _ %s\n",out->name,out->long_name,out->mime_type);
        //av_opt_show2((void *)&out->priv_class, NULL, -1, 0);
    }

    printf("-----\n");

    AVInputFormat *vi = NULL;
    while (vi = av_input_video_device_next(vi))
    {
        printf("%s,%s\n", vi->name, vi->long_name);
        
        AVDeviceInfoList *device_list = NULL;
        int err = avdevice_list_input_sources(NULL,vi->name,NULL,&device_list);
        if(err <0 || !device_list)
            continue;

        for (int i = 0; i < device_list->nb_devices; i++)
        {
            printf("%s,%s\n", device_list->devices[i]->device_name, device_list->devices[i]->device_description);

            AVFormatContext *ctx = abcdk_avformat_input_open(vi->name,device_list->devices[i]->device_name,NULL,NULL,NULL);

            AVOptionRanges *ranges = NULL;
            //av_opt_query_ranges(&ranges,(void *)&ctx->iformat->priv_class,"frame_size",AV_OPT_MULTI_COMPONENT_RANGE);
            av_opt_query_ranges(&ranges,(void *)&ctx->av_class,"list_formats",AV_OPT_MULTI_COMPONENT_RANGE);
            av_opt_freep_ranges(&ranges);

            abcdk_avformat_free(&ctx);
        }
        
        avdevice_free_list_devices(&device_list);
    }

    printf("-----\n");

    

#if 0    
    abcdk_image_t src = {AV_PIX_FMT_YUV420P,{NULL,NULL,NULL,NULL},{0,0,0,0},1920,1080};
    abcdk_image_t dst = {AV_PIX_FMT_YUV420P,{NULL,NULL,NULL,NULL},{0,0,0,0},1920,1080};
    abcdk_image_t dst2 = {AV_PIX_FMT_BGR32,{NULL,NULL,NULL,NULL},{0,0,0,0},1920,1080};

    int src_heights[4]={0}, dst_heights[4]={0}, dst2_heights[4]={0};


    abcdk_av_image_fill_heights(src_heights,src.height,src.pixfmt);
    abcdk_av_image_fill_heights(dst_heights,dst.height,dst.pixfmt);
    abcdk_av_image_fill_heights(dst2_heights,dst2.height,dst2.pixfmt);

    abcdk_av_image_fill_strides2(&src,16);
    abcdk_av_image_fill_strides2(&dst,10);
    abcdk_av_image_fill_strides2(&dst2,1);

    void *src_buf = abcdk_heap_alloc(abcdk_av_image_size3(&src));
    void *dst_buf = abcdk_heap_alloc(abcdk_av_image_size3(&dst));
    void *dst2_buf = abcdk_heap_alloc(abcdk_av_image_size3(&dst2));

    abcdk_av_image_fill_pointers2(&src,src_buf);
    abcdk_av_image_fill_pointers2(&dst,dst_buf);
    abcdk_av_image_fill_pointers2(&dst2,dst2_buf);

    abcdk_av_image_copy2(&dst,&src);

    struct SwsContext *ctx = abcdk_sws_alloc2(&src,&dst2,0);

    int h = sws_scale(ctx,(const uint8_t *const *)src.datas,src.strides,0,src.height,dst2.datas,dst2.strides);
    //int h = sws_scale(ctx,(const uint8_t *const *)src.datas,src.strides,100,src.height,dst2.datas,dst2.strides);

    printf("h = %d\n",h);

    uint8_t *tmp = dst2.datas[0];
    for (int i = 0; i < dst2.height; i++)
    {
        for (int j = 0; j < dst2.width*4; j += 4)
        {
            tmp[j+0] = 0;
            tmp[j+1] = 0;
            tmp[j+2] = 255;
        }

        tmp += dst2.strides[0];
    }

    int chk = abcdk_bmp_save2("/tmp/test_bmp.bmp",dst2.datas[0],dst2.strides[0],dst2.width,dst2.height,32);
    assert(chk==0);

    
    abcdk_sws_free(&ctx);

    abcdk_heap_free(src_buf);
    abcdk_heap_free(dst_buf);
    abcdk_heap_free(dst2_buf);

#endif

#endif //

}

void test_bmp(abcdk_tree_t *args)
{
    const char *src_file = abcdk_option_get(args,"--src-file",0,"");
    const char *dst_file = abcdk_option_get(args,"--dst-file",0,"");

    uint32_t stride = 0;
    uint32_t width = 0;
    int32_t height = 0;
    uint8_t bits = 0;
    int chk = abcdk_bmp_load2(src_file, NULL, 0, 13, &stride, &width, &height, &bits);
    assert(chk == 0);

    printf("s=%u,w=%u,h=%d,b=%hhu\n",stride,width,height,bits);

    uint8_t *data = abcdk_heap_alloc(stride*height);

    chk = abcdk_bmp_load2(src_file, data, stride*height, 1, &stride, &width, &height, &bits);
    assert(chk == 0);


    chk = abcdk_bmp_save2(dst_file, data, stride, width, height, bits);
    assert(chk == 0);


    abcdk_heap_free(data);
    
}

void test_freeimage(abcdk_tree_t *args)
{
#ifdef FREEIMAGE_H

    abcdk_fi_init(1);
    abcdk_fi_init(1);//test run once.

    abcdk_fi_log2syslog();

    const char *src_file = abcdk_option_get(args,"--src-file",0,"");
    const char *dst_file = abcdk_option_get(args,"--dst-file",0,"");

    uint8_t *data = NULL;
    uint32_t stride = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t bits = 0;
    uint32_t xbytes = 0;

    FREE_IMAGE_FORMAT src_fmt = FreeImage_GetFileType(src_file,0);

    FIBITMAP *dib = abcdk_fi_load2(src_fmt,0,src_file);
    assert(dib!=NULL);

    width = FreeImage_GetWidth(dib);
    height = FreeImage_GetHeight(dib);

    abcdk_resize_t r = {0};

    int dst_w = 500;
    int dst_h = 1100;

    abcdk_resize_ratio_2d(&r,width,height,dst_w,dst_h,0);

    FIBITMAP *dib2 = FreeImage_RescaleRect(dib,r.x_factor *width,r.y_factor*height,0,0,width,height,FILTER_BICUBIC,0);
    if(dib2)
    {
        FreeImage_Unload(dib);
        dib = dib2;
    }

    dib2 = FreeImage_ConvertTo24Bits(dib);
    if(dib2)
    {
        FreeImage_Unload(dib);
        dib = dib2;
    }

    int left = abcdk_resize_src2dst_2d(&r,0,1);
    int top = abcdk_resize_src2dst_2d(&r,0,0);
    dib2 = FreeImage_Allocate(dst_w,dst_h,24,0,0,0);
    FreeImage_Paste(dib2,dib,left,top,1000);
    if(dib2)
    {
        FreeImage_Unload(dib);
        dib = dib2;
    }

    data = FreeImage_GetBits(dib);
    stride = FreeImage_GetPitch(dib);
    width = FreeImage_GetWidth(dib);
    height = FreeImage_GetHeight(dib);
    bits = FreeImage_GetBPP(dib);
    xbytes = FreeImage_GetLine(dib);

   // FreeImage_FlipHorizontal(dib);
  //  FreeImage_FlipVertical(dib);

    //FreeImage_AdjustBrightness(dib,100);
    FreeImage_Invert(dib);
 
#if 1
    int chk = abcdk_fi_save2(FIF_JPEG,JPEG_QUALITYGOOD,dst_file, data, stride, width, height, bits);
    assert(chk == 0);
#else 
    BOOL chk = FreeImage_Save(FIF_JPEG,dib,dst_file,JPEG_QUALITYGOOD);
    assert(chk);
#endif 


    FreeImage_Unload(dib);


    abcdk_fi_uninit();
    abcdk_fi_uninit();//test run once.

#endif //FREEIMAGE_H
}

void test_uri(abcdk_tree_t *args)
{
    int n = abcdk_option_count(args, "--uri");
    for (int i = 0; i < n; i++)
    {
        const char *uri = abcdk_option_get(args, "--uri", i, "");

        abcdk_object_t *alloc = abcdk_uri_split(uri);
        assert(alloc);

        for (size_t i = 0; i < alloc->numbers; i++)
            printf("[%ld]: %s\n", i, alloc->pptrs[i]);

        abcdk_object_unref(&alloc);
    }
}

void test_strrep(abcdk_tree_t *args)
{
    char buf[]={"abcab|     |cabcabc"};

    char *p = abcdk_strrep(buf," ","",1);

    printf("%s\n",p);


    abcdk_heap_free(p);
}

/**/
const char *_test_html_cntrl_replace(char *text, char c)
{
    if(!text)
        return "";

    char *tmp = text;
    while (*tmp)
    {
        if (iscntrl(*tmp))
            *tmp = c;

        tmp += 1;
    }

    return text;
}

static int _test_html_dump_cb(size_t deep, abcdk_tree_t *node, void *opaque)
{
    if(deep==0)
    {
        abcdk_tree_fprintf(stderr,deep,node,"%s\n",".");
    }
    else
    {
            abcdk_tree_fprintf(stderr, deep, node, "%s:<%s>\n",
                               ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_HTML_KEY], 0),
                               _test_html_cntrl_replace(ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_HTML_VALUE], 0), ' '));
    }

    return 1;
}

void test_html(abcdk_tree_t *args)
{
    const char *file = abcdk_option_get(args,"--file",0,"");

 //   abcdk_clock_dot(NULL);

    abcdk_tree_t *t = abcdk_html_parse_file(file);

  //  printf("%lu\n",abcdk_clock_step(NULL));

    abcdk_tree_iterator_t it = {0,_test_html_dump_cb,NULL};

    abcdk_tree_scan(t,&it);

    abcdk_tree_free(&t);
}

void test_fnmatch(abcdk_tree_t *args)
{
 //   char str[]={"abcd?*Qcde"};
 //   char wd[]={"abc?\\?\\*q*****e"};

    char str[]={"/gp/aag/mainA?123456seller=ABVFEJU8LS620"};
    char wd[]={"/gp/aag/main\\?\\?*seller=ABVFEJU8LS620"};

    int chk = abcdk_fnmatch(str,wd,0,0);
    assert(chk==0);
}

void test_crc32(abcdk_tree_t *args)
{
//    uint32_t sum = abcdk_crc32_sum("abc",3,0);
//    printf("%u\n",sum);

    #pragma omp parallel for num_threads(30)
    for (int i = 0; i < 300000000; i++)
    {
        uint32_t sum2 = abcdk_crc32_sum("abc",3,0);
        assert(891568578 ==sum2);
    }
}

typedef struct _robots_match
{
    int flag;
    const char *path;
}robots_match_t;


static int _test_robots_dump_cb(size_t deep, abcdk_tree_t *node, void *opaque)
{
    if (deep == 0)
    {
        abcdk_tree_fprintf(stderr,deep, node, "%s\n", ".");
    }
    else
    {
        if (opaque)
        {
            robots_match_t *m = (robots_match_t*)opaque;

            int chk = abcdk_fnmatch(m->path,ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_VALUE], 0),0,0);
            if(chk==0)
            {
                if(abcdk_strcmp(ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_KEY], 0),"Disallow",0)==0)
                    m->flag = 2;
                if(abcdk_strcmp(ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_KEY], 0),"Allow",0)==0)
                    m->flag = 1;

            }
            else
            {
               // m->flag = -1;
            }
        }
        else
        {
            abcdk_tree_fprintf(stderr,deep, node, "%s: %s\n",
                               ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_KEY], 0),
                               ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_VALUE], 0));
        }
    }

    return 1;
}

void test_robots(abcdk_tree_t *args)
{
    const char *file = abcdk_option_get(args,"--file",0,"");
    const char *agent = abcdk_option_get(args,"--agent",0,"*");

    robots_match_t m = {0};
    m.path = abcdk_option_get(args,"--path",0,"");

    abcdk_tree_t *t = abcdk_robots_parse_file(file,agent);

    abcdk_tree_iterator_t it = {0,_test_robots_dump_cb,NULL};
    abcdk_tree_scan(t,&it);

    it.opaque = &m;
    abcdk_tree_scan(t,&it);

    printf("flag=%d\n",m.flag);

    abcdk_tree_free(&t);

}

#ifdef _FUSE_H_

#define MP4_PATH "/home/devel/job/tmp/"

/**/
int fuse_open(const char *file, struct fuse_file_info *info)
{
    syslog(LOG_INFO,"%s(%d): %s",__FUNCTION__,__LINE__,file);

    char tmp[PATH_MAX]={0};
    abcdk_dirdir(tmp,MP4_PATH);
    abcdk_dirdir(tmp,file);

    int fd = abcdk_open(tmp, 0, 0, 0);
    if (fd < 0)
        return -errno;

    info->fh = fd;
    info->direct_io = 1;
    info->keep_cache = 0;

    return 0;
}

int fuse_read(const char *file, char *buffer, size_t size, off_t offset, struct fuse_file_info *info)
{
    syslog(LOG_INFO, "%s(%d): %s (fd=%lu)", __FUNCTION__, __LINE__, file, info->fh);
    syslog(LOG_INFO, "%s(%d): size=%lu off=%ld", __FUNCTION__, __LINE__, size, offset);

    assert(info->fh != -1);

    int fd = info->fh;

    ssize_t rlen = pread(fd, buffer, size, offset);

    if(rlen != size)
        sleep(10);
    else 
        usleep(40*1000);

    return (rlen >= 0 ? rlen : -errno);
}

int fuse_release(const char* file, struct fuse_file_info *info)
{
    syslog(LOG_INFO, "%s(%d): %s (fd=%lu)", __FUNCTION__, __LINE__, file, info->fh);

    assert(info->fh != -1);

    int fd = info->fh;

    abcdk_closep(&fd);

    return 0;
}

int fuse_getattr(const char *file, struct stat* attr)
{
    syslog(LOG_INFO,"%s(%d): %s",__FUNCTION__,__LINE__,file);

    // if (abcdk_strcmp(file, "/") == 0)
    // {

    // }
    // else
    // {
        char tmp[PATH_MAX] = {0};
        abcdk_dirdir(tmp, MP4_PATH);
        abcdk_dirdir(tmp, file);

        int chk = lstat(tmp, attr);
        if (chk != 0)
            return -errno;

        attr->st_dev = 1000;
        clock_gettime(CLOCK_REALTIME, &attr->st_ctim);
        attr->st_mtim = attr->st_ctim;
        attr->st_size = INTMAX_MAX;
    // }

    return 0;
}

int fuse_fgetattr(const char* file, struct stat* attr, struct fuse_file_info * info)
{
    syslog(LOG_INFO, "%s(%d): %s (fd=%lu)", __FUNCTION__, __LINE__, file, info->fh);

    assert(info->fh != -1);

    int fd = info->fh;

    int chk = fstat(fd,attr);
    if(chk != 0 )
        return -errno;

     attr->st_dev = 1000;
     attr->st_size = INTMAX_MAX;

    return 0;
}

#endif //_FUSE_H_

void test_fuse(abcdk_tree_t *args)
{
#ifdef _FUSE_H_

    const char *name_p = abcdk_option_get(args,"--name",0,"test_fuse");
    const char *mpoint_p = abcdk_option_get(args,"--mpoint",0,"");

    if (strlen(name_p) <= 0)
    {
        syslog(LOG_ERR, "--name must have parameters.");
        return;
    }
    if (access(mpoint_p, R_OK) != 0)
    {
        syslog(LOG_ERR, "--mpoint must have parameters and exist.");
        return;
    }

    static struct fuse_operations opts = {0};
    opts.read = fuse_read;
    opts.open = fuse_open;
    opts.release = fuse_release;
    opts.getattr = fuse_getattr;
    opts.fgetattr = fuse_fgetattr;

    int fuse_argc = 4;
    char **fuse_argv = (char**)abcdk_heap_alloc(fuse_argc*sizeof(char*));

    fuse_argv[0] = abcdk_heap_clone(name_p,strlen(name_p));
    fuse_argv[1] = abcdk_heap_clone(mpoint_p,strlen(mpoint_p));
    fuse_argv[2] = abcdk_heap_clone("-o",2);
    fuse_argv[3] = abcdk_heap_clone("allow_other,auto_cache,kernel_cache",35);

    fuse_main(fuse_argc, fuse_argv, &opts, NULL);

#endif //_FUSE_H_
}

#if 0

int _mp4_read(abcdk_buffer_t *buf, void *data, size_t size)
{
    ssize_t r = abcdk_buffer_read(buf, data, size);
    if (r <= 0)
        return -2;
    else if (r != size)
        return -1;

    return 0;
}

int _mp4_read_u16(abcdk_buffer_t *buf, uint16_t *data)
{
    if (_mp4_read(buf, data, sizeof(uint16_t)))
        return -1;

    *data = abcdk_endian_b_to_h16(*data);

    return 0;
}

int _mp4_read_u24(abcdk_buffer_t *buf, uint8_t *data)
{
    if (_mp4_read(buf, data, sizeof(uint8_t)*3))
        return -1;

    abcdk_endian_b_to_h(data,3);

    return 0;
}

int _mp4_read_u32(abcdk_buffer_t *buf, uint32_t *data)
{
    if (_mp4_read(buf, data, sizeof(uint32_t)))
        return -1;

    *data = abcdk_endian_b_to_h32(*data);

    return 0;
}

int _mp4_read_u64(abcdk_buffer_t *buf,uint64_t *data)
{
    if (_mp4_read(buf, data, sizeof(uint64_t)))
        return -1;

    *data = abcdk_endian_b_to_h64(*data);

    return 0;
}

int _mp4_skip_size(abcdk_buffer_t *buf,uint64_t size)
{
    size_t all = 0;
    char tmp[1000];

    while(all<size)
    {
        size_t s = ABCDK_MIN(1000,size-all);
        if (_mp4_read(buf, tmp, s))
            return -1;

        all += s;
    }

    return 0;
}

void _mp4_dump_ftyp(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_ftyp_t *cont = (abcdk_mp4_atom_ftyp_t *)&atom->data;
    

    fprintf(stdout, "major='%c%c%c%c',", cont->major.u8[0], cont->major.u8[1], cont->major.u8[2], cont->major.u8[3] );
    fprintf(stdout, "minor='%d',", cont->minor);
    fprintf(stdout, "compatible=");
    for (size_t i = 0; i < cont->compat->numbers; i++)
    {
        abcdk_mp4_tag_t *brand = (abcdk_mp4_tag_t *)cont->compat->pptrs[i];
        if(!brand->u32)
            continue;
        fprintf(stdout, "'%c%c%c%c' ", brand->u8[0], brand->u8[1], brand->u8[2], brand->u8[3]);
    }

}

void _mp4_dump_mvhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_mvhd_t *cont = (abcdk_mp4_atom_mvhd_t *)&atom->data;

    fprintf(stdout, "version=%hhu,",cont->version);

    if(cont->ctime>=0x7C25B080)
        cont->ctime -= 0x7C25B080;

    struct tm t;
    gmtime_r(&cont->ctime,&t);
    fprintf(stdout, "ctime=%d-%02d-%02d %02d:%02d:%02d,",t.tm_year+1900,t.tm_mon+1,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec);
     
    if(cont->mtime>=0x7C25B080)
        cont->mtime -= 0x7C25B080;

    struct tm t2;
    gmtime_r(&cont->mtime,&t2);
    fprintf(stdout, "mtime=%d-%02d-%02d %02d:%02d:%02d,",t2.tm_year+1900,t2.tm_mon+1,t2.tm_mday,t2.tm_hour,t2.tm_min,t2.tm_sec);

    fprintf(stdout, "timescale=%u,",cont->timescale);
    fprintf(stdout, "duration=%lu,",cont->duration);
    fprintf(stdout, "rate=%hu.%hu,",cont->rate>>16,cont->rate&0xffff);
    fprintf(stdout, "long=%lu(sec),",cont->duration/cont->timescale);
    fprintf(stdout, "nexttrackid=%u,",cont->nexttrackid);
}

void _mp4_dump_tkhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_tkhd_t *cont = (abcdk_mp4_atom_tkhd_t *)&atom->data;

    fprintf(stdout, "version=%hhu,",cont->version);

    if(cont->ctime>=0x7C25B080)
        cont->ctime -= 0x7C25B080;

    struct tm t;
    gmtime_r(&cont->ctime,&t);
    fprintf(stdout, "ctime=%d-%02d-%02d %02d:%02d:%02d,",t.tm_year+1900,t.tm_mon+1,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec);
     
    if(cont->mtime>=0x7C25B080)
        cont->mtime -= 0x7C25B080;

    struct tm t2;
    gmtime_r(&cont->mtime,&t2);
    fprintf(stdout, "mtime=%d-%02d-%02d %02d:%02d:%02d,",t2.tm_year+1900,t2.tm_mon+1,t2.tm_mday,t2.tm_hour,t2.tm_min,t2.tm_sec);

    fprintf(stdout, "trackid=%u,",cont->trackid);
    fprintf(stdout, "duration=%lu,",cont->duration);
    fprintf(stdout, "width=%hu.%hu,",cont->width>>16,cont->width&0xffff);
    fprintf(stdout, "height=%hu.%hu,",cont->height>>16,cont->height&0xffff);
}

void _mp4_dump_mdhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_mdhd_t *cont = (abcdk_mp4_atom_mdhd_t *)&atom->data;

    fprintf(stdout, "version=%hhu,",cont->version);

    if(cont->ctime>=0x7C25B080)
        cont->ctime -= 0x7C25B080;

    struct tm t;
    gmtime_r(&cont->ctime,&t);
    fprintf(stdout, "ctime=%d-%02d-%02d %02d:%02d:%02d,",t.tm_year+1900,t.tm_mon+1,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec);
     
    if(cont->mtime>=0x7C25B080)
        cont->mtime -= 0x7C25B080;

    struct tm t2;
    gmtime_r(&cont->mtime,&t2);
    fprintf(stdout, "mtime=%d-%02d-%02d %02d:%02d:%02d,",t2.tm_year+1900,t2.tm_mon+1,t2.tm_mday,t2.tm_hour,t2.tm_min,t2.tm_sec);

    fprintf(stdout, "timescale=%u,",cont->timescale);
    fprintf(stdout, "duration=%lu,",cont->duration);
    fprintf(stdout, "lang=%hu,",cont->language);
    fprintf(stdout, "quality=%hu,",cont->quality);
}

void _mp4_dump_hdlr(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_hdlr_t *cont = (abcdk_mp4_atom_hdlr_t *)&atom->data;


    fprintf(stdout,"type=%c%c%c%c, ",
                           cont->type.u8[0], cont->type.u8[1], cont->type.u8[2], cont->type.u8[3]);

    fprintf(stdout,"subtype=%c%c%c%c, ",
                           cont->subtype.u8[0], cont->subtype.u8[1], cont->subtype.u8[2], cont->subtype.u8[3]);

    if(cont->name)
        fprintf(stdout,"name='%s' ",cont->name->pptrs[0]);
}


void _mp4_dump_vmhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_vmhd_t *cont = (abcdk_mp4_atom_vmhd_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "mode=%u,",cont->mode);
    fprintf(stdout, "opcolor=%hu,%hu,%hu",cont->opcolor[0],cont->opcolor[1],cont->opcolor[2]);
}


void _mp4_dump_stts(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stts_t *cont = (abcdk_mp4_atom_stts_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers;i++)
    {
        fprintf(stdout, "count=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));
        fprintf(stdout, "duration=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],4));
    }

}

void _mp4_dump_ctts(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_ctts_t *cont = (abcdk_mp4_atom_ctts_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "count=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));
        fprintf(stdout, "offset=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],4));
    }


}

void _mp4_dump_stsc(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stsc_t *cont = (abcdk_mp4_atom_stsc_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "Firstchunk=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));
        fprintf(stdout, "perchunk=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],4));
        fprintf(stdout, "ID=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],8));
    }
}

void _mp4_dump_stsz(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stsz_t *cont = (abcdk_mp4_atom_stsz_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "[samplesize=%u],",cont->samplesize);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));

    }

}


void _mp4_dump_stco(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stco_t *cont = (abcdk_mp4_atom_stco_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "%lu,",ABCDK_PTR2U64(cont->tables->pptrs[i],0));

    }

}

void _mp4_dump_stss(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stss_t *cont = (abcdk_mp4_atom_stss_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i < 10;i++)
    {
        fprintf(stdout, "%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));

    }

}

void _mp4_dump_smhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_smhd_t *cont = (abcdk_mp4_atom_smhd_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "balance=%hhu.%hhu",cont->balance>>8,cont->balance&0xff);
}

void _mp4_dump_elst(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_elst_t *cont = (abcdk_mp4_atom_elst_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "duration=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));
        fprintf(stdout, "time=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],4));
        fprintf(stdout, "rate=%hu.%hu,",
            ABCDK_PTR2U32(cont->tables->pptrs[i],8)>>16,
            ABCDK_PTR2U32(cont->tables->pptrs[i],8)&&0xffff);
    }

}

void _mp4_dump_mehd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_mehd_t *cont = (abcdk_mp4_atom_mehd_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "duration=%lu",cont->duration);
}

void _mp4_dump_trex(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_trex_t *cont = (abcdk_mp4_atom_trex_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "trackid=%u,",cont->trackid);
    fprintf(stdout, "sample_desc_index=%u,",cont->default_sample_desc_index);
    fprintf(stdout, "duration=%lu,",cont->default_duration);
    fprintf(stdout, "sample_sample_size=%u,",cont->default_samplesize);
    fprintf(stdout, "sample_flags=%08x",cont->default_sampleflags);
}

void _mp4_dump_mfhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_mfhd_t *cont = (abcdk_mp4_atom_mfhd_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "sn=%lu,",cont->sn);

}

void _mp4_dump_tfhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_tfhd_t *cont = (abcdk_mp4_atom_tfhd_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "trackid=%u,",cont->trackid);
    fprintf(stdout, "base_data_offset=%lu,",cont->base_data_offset);
    fprintf(stdout, "sample_desc_index=%u,",cont->sample_desc_index);
    fprintf(stdout, "duration=%lu,",cont->default_duration);
    fprintf(stdout, "sample_sample_size=%u,",cont->default_samplesize);
    fprintf(stdout, "sample_flags=%08x",cont->default_sampleflags);
}

void _mp4_dump_tfdt(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_tfdt_t *cont = (abcdk_mp4_atom_tfdt_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "time=%lu,",cont->base_decode_time);

}

void _mp4_dump_trun(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_trun_t *cont = (abcdk_mp4_atom_trun_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "offset=%u,",cont->data_offset);
    fprintf(stdout, "flags=%08x,",cont->first_sample_flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers;i++)
    {
        fprintf(stdout, "[%ld]={",i);
        fprintf(stdout, "duration=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));
        fprintf(stdout, "size=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],4));
        fprintf(stdout, "flags=%08x,",ABCDK_PTR2U32(cont->tables->pptrs[i],8));
        fprintf(stdout, "offset=%u",ABCDK_PTR2U32(cont->tables->pptrs[i],12));
        fprintf(stdout, "},");
     }
}

void _mp4_dump_mfro(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_mfro_t *cont = (abcdk_mp4_atom_mfro_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "size=%lu,",cont->size);

}


void _mp4_dump_tfra(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_tfra_t *cont = (abcdk_mp4_atom_tfra_t *)&atom->data;

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "trackid=%u,",cont->trackid);
    fprintf(stdout, "size_traf_num=%hhu,",cont->length_size_traf_num);
    fprintf(stdout, "size_trun_num=%hhu,",cont->length_size_trun_num);
    fprintf(stdout, "size_sample_num=%hhu,",cont->length_size_sample_num);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers;i++)
    {
        fprintf(stdout, "[%ld]={",i);
        fprintf(stdout, "time=%lu,",ABCDK_PTR2U64(cont->tables->pptrs[i],0));
        fprintf(stdout, "moof offset=%lu,",ABCDK_PTR2U64(cont->tables->pptrs[i],8));
        fprintf(stdout, "traf=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],16));
        fprintf(stdout, "trun=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],20));
        fprintf(stdout, "sample=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],24));
        fprintf(stdout, "},");
     }
}

static int atoms =0;

int mp4_dump_cb(size_t deep, abcdk_tree_t *node, void *opaque)
{
    if (deep == -1)
        return -1;

    atoms += 1;

    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];

    if (deep == 0)
    {
        abcdk_tree_fprintf(stdout, deep, node, ".\n");
    }
    else
    {
        abcdk_tree_fprintf(stdout, deep, node, "offset=%lu,size=%lu,type=%c%c%c%c: ",
                           atom->off_head, atom->size, atom->type.u8[0], atom->type.u8[1], atom->type.u8[2], atom->type.u8[3]);

        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_FTYP)
            _mp4_dump_ftyp(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MVHD)
            _mp4_dump_mvhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TKHD)
            _mp4_dump_tkhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MDHD)
            _mp4_dump_mdhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HDLR)
            _mp4_dump_hdlr(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_VMHD)
            _mp4_dump_vmhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STTS)
            _mp4_dump_stts(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_CTTS)
            _mp4_dump_ctts(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSC)
            _mp4_dump_stsc(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSZ)
            _mp4_dump_stsz(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STCO)
            _mp4_dump_stco(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSS)
            _mp4_dump_stss(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_SMHD)
            _mp4_dump_smhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_ELST)
            _mp4_dump_elst(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MEHD)
            _mp4_dump_mehd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TREX)
            _mp4_dump_trex(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MFHD)
            _mp4_dump_mfhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TFHD)
            _mp4_dump_tfhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TFDT)
            _mp4_dump_tfdt(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TRUN)
            _mp4_dump_trun(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MFRO)
            _mp4_dump_mfro(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TFRA)
            _mp4_dump_tfra(deep, node, opaque);



        fprintf(stdout, " \n");

   //     if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MOOF)
   //         return -1;


    }

 //  if(atoms>70)
 //       return -1;

    return 1;
}

#endif 

void show_mp4_info(int fd)
{
    abcdk_tree_t *root = abcdk_mp4_read_probe(fd,0,-1UL, NULL);
   
    abcdk_tree_t *video_p = abcdk_mp4_find2(root,ABCDK_MP4_ATOM_TYPE_TRAK,1,1);
    abcdk_tree_t *avc1_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_AVC1,1,1);
    abcdk_tree_t *avcc_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_AVCC,1,1);

    abcdk_mp4_atom_t *avc1 = (abcdk_mp4_atom_t*)avc1_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *avcc = (abcdk_mp4_atom_t*)avcc_p->alloc->pptrs[0];
#ifdef HAVE_FFMPEG
    AVCodecContext *enc_ctx = abcdk_avcodec_alloc(abcdk_avcodec_find2(AV_CODEC_ID_H264,0));

    enc_ctx->extradata_size = avcc->data.avcc.extradata->sizes[0];
    enc_ctx->extradata = av_mallocz(avcc->data.avcc.extradata->sizes[0]);
    memcpy(enc_ctx->extradata,avcc->data.avcc.extradata->pptrs[0],avcc->data.avcc.extradata->sizes[0]);

    enc_ctx->width = avc1->data.sample_desc.detail.video.width;
    enc_ctx->height = avc1->data.sample_desc.detail.video.height;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;

    assert(abcdk_avcodec_open(enc_ctx,NULL)==0);

    AVFrame *frame_p = av_frame_alloc();
    AVPacket packet = {0};
    av_init_packet(&packet);

    packet.data = 0;
    packet.size = 0;
    packet.stream_index = 0;

    assert(abcdk_avcodec_decode(enc_ctx,frame_p,&packet)>=0);
    
    av_frame_free(&frame_p);
    av_packet_unref(&packet);

    abcdk_avcodec_free(&enc_ctx);

#endif //HAVE_FFMPEG

    abcdk_tree_free(&root);
}

void collect_fmp4_video(int fd)
{
    int fd2 = abcdk_open("/tmp/abcdk2.h264",1,0,1);
    // ftruncate(fd2,0);
    lseek(fd2,0,SEEK_END);

    abcdk_mp4_tag_t a;
    a.u32 = ABCDK_MP4_ATOM_MKTAG('\0','\0','\0','\1');

    char *buf= abcdk_heap_alloc(1024*1024*16);

    abcdk_tree_t *root = abcdk_mp4_read_probe2(fd,0,-1UL, 0);

    abcdk_tree_t *moov_p = abcdk_mp4_find2(root,ABCDK_MP4_ATOM_TYPE_MOOV,1,1);
    abcdk_tree_t *mvex_p = abcdk_mp4_find2(root,ABCDK_MP4_ATOM_TYPE_MVEX,1,1);

    abcdk_mp4_dump(stdout,moov_p);

    abcdk_tree_t *avcc_p = abcdk_mp4_find2(moov_p,ABCDK_MP4_ATOM_TYPE_AVCC,1,1);
    abcdk_mp4_atom_t *avcc = (abcdk_mp4_atom_t*)avcc_p->alloc->pptrs[0];

    abcdk_tree_t *mehd_p = abcdk_mp4_find2(mvex_p,ABCDK_MP4_ATOM_TYPE_MEHD,1,1);
    abcdk_mp4_atom_t *mehd = (abcdk_mp4_atom_t*)mehd_p->alloc->pptrs[0];

    abcdk_tree_t *moof_p = abcdk_tree_child(root,1);
    while (moof_p)
    {
        abcdk_mp4_atom_t *moof = (abcdk_mp4_atom_t*)moof_p->alloc->pptrs[0];
        if(moof->type.u32 == ABCDK_MP4_ATOM_TYPE_MOOF)
        {
            abcdk_tree_t *mfhd_p = abcdk_mp4_find2(moof_p, ABCDK_MP4_ATOM_TYPE_MFHD, 1, 1);
            abcdk_tree_t *tfhd_p = abcdk_mp4_find2(moof_p, ABCDK_MP4_ATOM_TYPE_TFHD, 1, 1);
            abcdk_tree_t *tfdt_p = abcdk_mp4_find2(moof_p, ABCDK_MP4_ATOM_TYPE_TFDT, 1, 1);
            abcdk_tree_t *trun_p = abcdk_mp4_find2(moof_p, ABCDK_MP4_ATOM_TYPE_TRUN, 1, 1);

            abcdk_mp4_atom_t *mfhd = (abcdk_mp4_atom_t *)mfhd_p->alloc->pptrs[0];
            abcdk_mp4_atom_t *tfhd = (abcdk_mp4_atom_t *)tfhd_p->alloc->pptrs[0];
            abcdk_mp4_atom_t *tfdt = (abcdk_mp4_atom_t *)tfdt_p->alloc->pptrs[0];
            abcdk_mp4_atom_t *trun = (abcdk_mp4_atom_t *)trun_p->alloc->pptrs[0];
#if 1
            printf("-----------------------------------mfhd---------------------------------------\n");

            printf("Sequence_Number: %lu\n", mfhd->data.mfhd.sequence_number);

            printf("-----------------------------------mfhd---------------------------------------\n");

            printf("-----------------------------------tfhd---------------------------------------\n");

            printf("TrackID: %u\n", tfhd->data.tfhd.trackid);
            printf("Base_Data_Offset: %lu\n", tfhd->data.tfhd.base_data_offset);
            printf("Sample_Desc_Index: %u\n", tfhd->data.tfhd.sample_desc_idx);

            printf("-----------------------------------tfhd---------------------------------------\n");

            printf("-----------------------------------tfdt---------------------------------------\n");

            printf("base_decode_time: %lu\n", tfdt->data.tfdt.base_decode_time);

            printf("-----------------------------------tfdt---------------------------------------\n");

            printf("-----------------------------------trun---------------------------------------\n");

            printf("Data_Offset: %u\n", trun->data.trun.data_offset);
            printf("First_Sample_Flags: %08x\n", trun->data.trun.first_sample_flags);
            printf("Numbers: %u\n", trun->data.trun.numbers);

            uint64_t duration_start = tfdt->data.tfdt.base_decode_time;
            for (size_t i = 0; i < trun->data.trun.numbers; i++)
            {
                

                uint64_t duration = tfhd->data.tfhd.sample_duration;
                
                duration = tfhd->data.tfhd.sample_duration;
                if(trun->data.trun.flags & ABCDK_MP4_TRUN_FLAG_SAMPLE_DURATION_PRESENT)
                    duration =  trun->data.trun.tables[i].sample_duration;

                duration_start += duration;
                printf("Size: %u,PTS: %lu(%lu+%d), DUR: %lu\n", 
                trun->data.trun.tables[i].sample_size,
                duration_start+trun->data.trun.tables[i].composition_offset,
                duration_start,
                trun->data.trun.tables[i].composition_offset,
                duration);
                
   

            }

            printf("-----------------------------------trun---------------------------------------\n");
#else
            if (tfhd->data.tfhd.trackid == 1)
            {

                lseek(fd, moof->off_head + trun->data.trun.data_offset, SEEK_SET);

                for (size_t i = 0; i < trun->data.trun.numbers; i++)
                {
                    abcdk_mp4_read(fd, buf, trun->data.trun.tables[i].sample_size);

                    abcdk_write(fd2, &a.u32, 4);
                    abcdk_write(fd2, avcc->data.avcc.sps->pptrs[0], avcc->data.avcc.sps->sizes[0]);
                    abcdk_write(fd2, &a.u32, 4);
                    abcdk_write(fd2, avcc->data.avcc.pps->pptrs[0], avcc->data.avcc.pps->sizes[0]);
                    memcpy(buf, &a.u32, 4); //替换长度
                    abcdk_write(fd2, buf, trun->data.trun.tables[i].sample_size);
                }
            }
#endif
        }

        moof_p = abcdk_tree_sibling(moof_p,0);
    }
    
    abcdk_heap_free(buf);
    abcdk_tree_free(&root);
    abcdk_closep(&fd2);
}

void collect_mp4_video(int fd)
{


    abcdk_tree_t *root = abcdk_mp4_read_probe2(fd,0,-1UL, 0);

    abcdk_mp4_dump(stdout,root);

    abcdk_tree_t *video_p = abcdk_mp4_find2(root,ABCDK_MP4_ATOM_TYPE_TRAK,1,1);
    abcdk_tree_t *stsz_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STSZ,1,1);
    abcdk_tree_t *stss_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STSS,1,1);
    abcdk_tree_t *stts_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STTS,1,1);
    abcdk_tree_t *ctts_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_CTTS,1,1);
    abcdk_tree_t *stsc_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STSC,1,1);
    abcdk_tree_t *stco_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STCO,1,1);
    abcdk_tree_t *avc1_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_AVC1,1,1);
    abcdk_tree_t *avcc_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_AVCC,1,1);

    abcdk_mp4_atom_t *stsz = (abcdk_mp4_atom_t*)stsz_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *stss = (abcdk_mp4_atom_t*)stss_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *stts = (abcdk_mp4_atom_t*)stts_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *ctts = (abcdk_mp4_atom_t*)ctts_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *stco = (abcdk_mp4_atom_t*)stco_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *stsc = (abcdk_mp4_atom_t*)stsc_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *avc1 = (abcdk_mp4_atom_t*)avc1_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *avcc = (abcdk_mp4_atom_t*)avcc_p->alloc->pptrs[0];

    char sps[200] = {0};
    abcdk_bin2hex(sps,avcc->data.avcc.sps->pptrs[0],avcc->data.avcc.sps->sizes[0],0);
    printf("SPS:[%s]\n",sps);

    char pps[200] = {0};
    abcdk_bin2hex(pps,avcc->data.avcc.pps->pptrs[0],avcc->data.avcc.pps->sizes[0],0);
    printf("PPS:[%s]\n",pps);

#if 0


    printf("-----------------------------------stsz---------------------------------------\n");
    printf("Size: %u\n",stsz->data.stsz.sample_size);
    printf("Numbers: %u\n",stsz->data.stsz.numbers);
    for (size_t i = 0; i < stsz->data.stsz.numbers; i++)
    {
        uint64_t dts = 0;
        uint32_t dur = 0;
        int32_t cts = 0;
        abcdk_mp4_stts_tell(&stts->data.stts,i+1,&dts,&dur);
        abcdk_mp4_ctts_tell(&ctts->data.ctts,i+1,&cts);
        printf("Size[%lu]: %u, PTS: %lu(%lu+%d) DUR: %u, KEY: %s\n",
                    i+1,stsz->data.stsz.tables[i].size,dts+cts,dts,cts,dur,
                    (abcdk_mp4_stss_tell(&stss->data.stss,i+1)?"No":"Yes") );
    }
    printf("-----------------------------------stsz---------------------------------------\n");

    printf("-----------------------------------stss---------------------------------------\n");
    printf("Numbers: %u\n",stss->data.stss.numbers);
    for (size_t i = 0; i < stss->data.stss.numbers; i++)
    {
        printf("KeyFrame[%lu]: %u\n",i+1,stss->data.stss.tables[i].sync);
    }
    printf("-----------------------------------stss---------------------------------------\n");

    printf("-----------------------------------stts---------------------------------------\n");
    printf("Numbers: %u\n",stts->data.stts.numbers);
    for (size_t i = 0; i < stts->data.stts.numbers; i++)
    {
        printf("Count[%lu]: %u\n",i+1,stts->data.stts.tables[i].sample_count);
        printf("Duration[%lu]: %u\n",i+1,stts->data.stts.tables[i].sample_duration);
    }
    printf("-----------------------------------stts---------------------------------------\n");

     printf("-----------------------------------ctts---------------------------------------\n");
    printf("Numbers: %u\n",ctts->data.ctts.numbers);
    for (size_t i = 0; i < ctts->data.ctts.numbers; i++)
    {
        printf("Count[%lu]: %u\n",i+1,ctts->data.ctts.tables[i].sample_count);
        printf("Offset[%lu]: %u\n",i+1,ctts->data.ctts.tables[i].composition_offset);
    }
    printf("-----------------------------------ctts---------------------------------------\n");

    printf("-----------------------------------stco---------------------------------------\n");
    printf("Numbers: %u\n",stco->data.stco.numbers);
    for (size_t i = 0; i < stco->data.stco.numbers; i++)
    {
        printf("Offset[%lu]: %lu\n",i+1,stco->data.stco.tables[i].offset);
    }
    printf("-----------------------------------stco---------------------------------------\n");

    printf("-----------------------------------stsc---------------------------------------\n");
    printf("Numbers: %u\n",stsc->data.stsc.numbers);
    for(size_t i= 0 ;i<stsc->data.stsc.numbers;i++)
    {
        printf("First_Chunk: %u\n",stsc->data.stsc.tables[i].first_chunk);
        printf("PerChunk: %u\n",stsc->data.stsc.tables[i].samples_perchunk);
        printf("ID: %u\n",stsc->data.stsc.tables[i].sample_desc_id);
    }
    printf("-----------------------------------stsc---------------------------------------\n");

    

#else 


    int fd2 = abcdk_open("/tmp/abcdk.h264",1,0,1);
   // ftruncate(fd2,0);
   lseek(fd2,0,SEEK_END);

  //  abcdk_write(fd2,avcc->data.avcc.extradata->pptrs[0],avcc->data.avcc.extradata->sizes[0]);

    char *buf= abcdk_heap_alloc(1024*1024*16);

    abcdk_mp4_tag_t a;
    a.u32 = ABCDK_MP4_ATOM_MKTAG('\0','\0','\0','\1');


    for(size_t i = 1 ;i<=stsz->data.stsz.numbers;i++)
    {
        uint32_t chunk=0, offset=0, id=0;
        abcdk_mp4_stsc_tell(&stsc->data.stsc,i,&chunk,&offset,&id);

        printf("[%lu]={chunk=%u,offset=%u,id=%u}\n",i,chunk,offset,id);

        uint32_t offset2=0, size = 0;
        abcdk_mp4_stsz_tell(&stsz->data.stsz,offset,i,&offset2,&size);

        printf("[%lu]={offset2=%u,size=%u}\n",i,offset2,size);

        lseek(fd,stco->data.stco.tables[chunk-1].offset + offset2,SEEK_SET);

        abcdk_mp4_read(fd,buf,size);

        abcdk_write(fd2,&a.u32,4);
        abcdk_write(fd2,avcc->data.avcc.sps->pptrs[0],avcc->data.avcc.sps->sizes[0]);
        abcdk_write(fd2,&a.u32,4);
        abcdk_write(fd2,avcc->data.avcc.pps->pptrs[0],avcc->data.avcc.pps->sizes[0]);
        memcpy(buf,&a.u32,4);//替换长度
        abcdk_write(fd2,buf,size);
    }

    abcdk_closep(&fd2);
    abcdk_heap_free(buf);
#endif
    
    abcdk_tree_free(&root);
}

#define ADTS_HEADER_SIZE 7

typedef struct _adtsctx
{
    int write_adts;
    int objecttype;
    int sample_rate_index;
    int channel_conf;
} adtsctx;

int aac_decode_extradata(adtsctx *adts, unsigned char *pbuf, int bufsize)
{
    int aot, aotext, samfreindex;
    int i, channelconfig;
    unsigned char *p = pbuf;
    if (!adts || !pbuf || bufsize < 2)
    {
        return -1;
    }
    aot = (p[0] >> 3) & 0x1f;
    if (aot == 31)
    {
        aotext = (p[0]<<3 | (p[1]>>5)) & 0x3f;
        aot = 32 + aotext;
        samfreindex = (p[1] >> 1) & 0x0f;
        if (samfreindex == 0x0f)
        {

            channelconfig = ((p[4] << 3) | (p[5] >> 5)) & 0x0f;
        }
        else
        {

            channelconfig = ((p[1] << 3) | (p[2] >> 5)) & 0x0f;
        }
    }
    else
    {
        samfreindex = ((p[0] << 1) | p[1] >> 7) & 0x0f;
        if (samfreindex == 0x0f)
        {
            channelconfig = (p[4] >> 3) & 0x0f;
        }
        else
        {

            channelconfig = (p[1] >> 3) & 0x0f;
        }
    }
#ifdef AOT_PROFILE_CTRL

    if (aot < 2)
        aot = 2;
#endif
    
    adts->objecttype = aot-1;
    adts->sample_rate_index = samfreindex;
    adts->channel_conf = channelconfig;
    adts->write_adts = 1;

    return 0;
}

int aac_set_adts_head(adtsctx *acfg, unsigned char *buf, int size)
{
    unsigned char byte;
    if (size < ADTS_HEADER_SIZE)
        return -1;

    buf[0] = 0xff;

    buf[1] = 0xf1;
    byte = 0;
    byte |= (acfg->objecttype & 0x03) << 6;
    byte |= (acfg->sample_rate_index & 0x0f) << 2;
    byte |= (acfg->channel_conf & 0x07) >> 2;

    buf[2] = byte;
    byte = 0;
    byte |= (acfg->channel_conf & 0x07) << 6;
    byte |= (ADTS_HEADER_SIZE + size) >> 11;

    buf[3] = byte;
    byte = 0;
    byte |= (ADTS_HEADER_SIZE + size) >> 3;

    buf[4] = byte;
    byte = 0;
    byte |= ((ADTS_HEADER_SIZE + size) & 0x7) << 5;
    byte |= (0x7ff >> 6) & 0x1f;

    buf[5] = byte;
    byte = 0;
    byte |= (0x7ff & 0x3f) << 2;

    buf[6] = byte;

    return 0;
}

void collect_mp4_sound(int fd)
{

    abcdk_tree_t *root = abcdk_mp4_read_probe2(fd,0,-1UL, 0);

    abcdk_mp4_dump(stdout,root);

    abcdk_tree_t *video_p = abcdk_mp4_find2(root,ABCDK_MP4_ATOM_TYPE_TRAK,2,1);
    abcdk_tree_t *stsz_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STSZ,1,1);
    abcdk_tree_t *stss_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STSS,1,1);
    abcdk_tree_t *stts_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STTS,1,1);
    abcdk_tree_t *ctts_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_CTTS,1,1);
    abcdk_tree_t *stsc_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STSC,1,1);
    abcdk_tree_t *stco_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_STCO,1,1);
    abcdk_tree_t *mp4a_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_MP4A,1,1);
    abcdk_tree_t *esds_p = abcdk_mp4_find2(video_p,ABCDK_MP4_ATOM_TYPE_ESDS,1,1);

    abcdk_mp4_atom_t *stsz = (abcdk_mp4_atom_t*)stsz_p->alloc->pptrs[0];
   // abcdk_mp4_atom_t *stss = (abcdk_mp4_atom_t*)stss_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *stts = (abcdk_mp4_atom_t*)stts_p->alloc->pptrs[0];
   // abcdk_mp4_atom_t *ctts = (abcdk_mp4_atom_t*)ctts_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *stco = (abcdk_mp4_atom_t*)stco_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *stsc = (abcdk_mp4_atom_t*)stsc_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *mp4a = (abcdk_mp4_atom_t*)mp4a_p->alloc->pptrs[0];
    abcdk_mp4_atom_t *esds = (abcdk_mp4_atom_t*)esds_p->alloc->pptrs[0];



    int fd2 = abcdk_open("/tmp/abcdk.acc",1,0,1);
    // ftruncate(fd2,0);
    lseek(fd2,0,SEEK_END);

    char *buf= abcdk_heap_alloc(1024*1024*16);

    adtsctx adts={0};
    aac_decode_extradata(&adts,esds->data.esds.dec_sp_info.extradata->pptrs[0],esds->data.esds.dec_sp_info.extradata->sizes[0]);

    for(size_t i = 1 ;i<=stsz->data.stsz.numbers;i++)
    {
        uint32_t chunk=0, offset=0, id=0;
        abcdk_mp4_stsc_tell(&stsc->data.stsc,i,&chunk,&offset,&id);

        printf("[%lu]={chunk=%u,offset=%u,id=%u}\n",i,chunk,offset,id);

        uint32_t offset2=0, size = 0;
        abcdk_mp4_stsz_tell(&stsz->data.stsz,offset,i,&offset2,&size);

        printf("[%lu]={offset2=%u,size=%u}\n",i,offset2,size);

        lseek(fd,stco->data.stco.tables[chunk-1].offset + offset2,SEEK_SET);

        abcdk_mp4_read(fd,buf,size);

        char hdr[7]={0};
        aac_set_adts_head(&adts,hdr,size);
        abcdk_write(fd2,hdr,7);
        abcdk_write(fd2,buf,size);

    }

    abcdk_closep(&fd2);
    abcdk_heap_free(buf);
    abcdk_tree_free(&root);
}

void test_mp4(abcdk_tree_t *args)
{
    const char *name_p = abcdk_option_get(args,"--file",0,"");

#if 0

    abcdk_object_t *t = abcdk_mmap2(name_p,0,0);
    if(!t)
        return;

    abcdk_buffer_t *buf = abcdk_buffer_alloc(t);
    if(!buf)
    {
        abcdk_object_unref(&t);
        return;
    }

    buf->wsize = t->sizes[0];

    while (1)
    {
        uint32_t size2 = 0;
        uint64_t size = 0;
        if (_mp4_read_u32(buf, &size2))
            break;

        uint32_t type = 0;
        if (_mp4_read(buf, &type, sizeof(uint32_t)))
            break;

        for (int i = 0; i < 4; i++)
            printf("%c", ABCDK_PTR2I8(&type, i));
        printf("\n");

        if (size2 == 0)
            break;
        else if (size2 == 1)
        {
            if (_mp4_read_u64(buf, &size))
                break;
        }
        
        size = size2;
        size_t hsize = (size2==1?16:8);
        
        /*skip data*/
        if(_mp4_skip_size(buf,size-hsize))
            break;
    }

    abcdk_buffer_free(&buf);

#else 
    int fd = abcdk_open(name_p,0,0,0);
    if(fd<0)
        return;

#if 0

    abcdk_tree_t *root = abcdk_mp4_read_probe(fd,0,-1UL, NULL);

    abcdk_tree_t *ftyp = abcdk_mp4_find2(root,ABCDK_MP4_ATOM_TYPE_FTYP,1,0);
    abcdk_tree_t *moov = abcdk_mp4_find2(root,ABCDK_MP4_ATOM_TYPE_MOOV,1,0);
  
     
    abcdk_tree_iterator_t it = {0,mp4_dump_cb,(void*)(int64_t)fd};
    abcdk_tree_scan(root,&it);

    printf("\natoms:%d\n",atoms);


    abcdk_tree_free(&root);
#else 

  //  show_mp4_info(fd);

    //collect_mp4_video(fd);
    collect_mp4_sound(fd);

    //collect_fmp4_video(fd);

#endif 

    abcdk_closep(&fd);

#endif 
}

void test_dirent(abcdk_tree_t *args)
{
    const char *path_p = abcdk_option_get(args,"--path",0,"");

    abcdk_tree_t *t = abcdk_tree_alloc3(1);

    abcdk_dirent_open(t,path_p);

    for(;;)
    {
        char file[PATH_MAX]={0};
        int chk = abcdk_dirent_read(t,file);
        if(chk != 0)
            break;
        
        printf("%s\n",file);
        
 //       abcdk_dirent_open(t,file);
    }

}

void test_netlink(abcdk_tree_t *args)
{
    const char *ap = abcdk_option_get(args,"--i",0,"");

    int flag = 0;

    int chk = abcdk_netlink_fetch(ap,&flag);

    if (chk == 0)
    {
        printf("%s: UP=%s,BCAST=%s,MCAST=%s,LOOP=%s,P2P=%s,RUN=%s\n", ap,
               (flag & IFF_UP) ? "Yes" : "No",
               (flag & IFF_BROADCAST) ? "Yes" : "No",
               (flag & IFF_MULTICAST) ? "Yes" : "No",
               (flag & IFF_LOOPBACK) ? "Yes" : "No",
               (flag & IFF_POINTOPOINT) ? "Yes" : "No",
               (flag & IFF_RUNNING) ? "Yes" : "No");
    }
    else
        printf("%s: %s\n", ap, strerror(errno));
}

#ifdef HAVE_LIBNM
void request_rescan_cb (GObject *object, GAsyncResult *result, gpointer user_data)
{
	NMClient *cli = (NMClient *) user_data;
	GError *error = NULL;

	nm_device_wifi_request_scan_finish (NM_DEVICE_WIFI (object), result, &error);
	// if (error) {
	// 	g_string_printf (nmc->return_text, _("Error: %s."), error->message);
	// 	nmc->return_value = NMC_RESULT_ERROR_UNKNOWN;
	// 	g_error_free (error);
	// }
}

static int
compare_devices (const void *a, const void *b)
{
	NMDevice *da = *(NMDevice **)a;
	NMDevice *db = *(NMDevice **)b;
	int cmp;

	/* Sort by later device states first */
	cmp = nm_device_get_state (db) - nm_device_get_state (da);
	if (cmp != 0)
		return cmp;

	cmp = g_strcmp0 (nm_device_get_type_description (da),
	                 nm_device_get_type_description (db));
	if (cmp != 0)
		return cmp;

	return g_strcmp0 (nm_device_get_iface (da),
	                  nm_device_get_iface (db));
}

static NMDevice **
get_devices_sorted (NMClient *client)
{
	const GPtrArray *devs;
	NMDevice **sorted;

	devs = nm_client_get_devices (client);

	sorted = g_new (NMDevice *, devs->len + 1);
	memcpy (sorted, devs->pdata, devs->len * sizeof (NMDevice *));
	sorted[devs->len] = NULL;

	qsort (sorted, devs->len, sizeof (NMDevice *), compare_devices);
	return sorted;
}

#endif //HAVE_LIBNM

void
iw_essid_escape(char *		dest,
		const char *	src,
		const int	slen)
{
  const unsigned char *	s = (const unsigned char *) src;
  const unsigned char *	e = s + slen;
  char *		d = dest;

  /* Look every character of the string */
  while(s < e)
    {
      int	isescape;

      /* Escape the escape to avoid ambiguity.
       * We do a fast path test for performance reason. Compiler will
       * optimise all that ;-) */
      if(*s == '\\')
	{
	  /* Check if we would confuse it with an escape sequence */
	  if((e-s) > 4 && (s[1] == 'x')
	     && (isxdigit(s[2])) && (isxdigit(s[3])))
	    {
	      isescape = 1;
	    }
	  else
	    isescape = 0;
	}
      else
	isescape = 0;
      

      /* Is it a non-ASCII character ??? */
      if(isescape || !isascii(*s) || iscntrl(*s))
	{
	  /* Escape */
	  sprintf(d, "\\x%02X", *s);
	  d += 4;
	}
      else
	{
	  /* Plain ASCII, just copy */
	  *d = *s;
	  d++;
	}
      s++;
    }

  /* NUL terminate destination */
  *d = '\0';
}


void test_iwscan(abcdk_tree_t *args)
{
#if 0

    abcdk_object_t * k = abcdk_object_alloc(NULL,1,0);
    abcdk_object_t * p = abcdk_object_alloc(NULL,1,0);

    k->pptrs[0] = "GH";
    k->sizes[0] = 2;

    p->pptrs[0] = ABCDK_ANSI_COLOR_RED;
    
    //

    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    struct	iw_scan_req scan_req = {0};
   // scan_req.scan_type = IW_SCAN_TYPE_ACTIVE;
    //scan_req.flags = ;

    struct  iwreq req = {0};
    strncpy(req.ifr_ifrn.ifrn_name, "wlx70f11c3c3500", IFNAMSIZ);
    //req.u.data.pointer = &scan_req;
    //req.u.data.length = sizeof(struct	iw_scan_req);
    //req.u.data.flags = IW_SCAN_DEFAULT;

    
   // int sock = socket(AF_INET, SOCK_DGRAM, 0);

 //   int chk = abcdk_socket_ioctl(SIOCSIWSCAN,&req);
    int chk = ioctl(sock, SIOCSIWSCAN,&req);

  abcdk_object_t * scan_rsp = abcdk_object_alloc2(100000);

    /* Forever */
    while (1)
    {
        
            struct  iwreq rsp = {0};
            strncpy(rsp.ifr_ifrn.ifrn_name, "wlx70f11c3c3500", IFNAMSIZ);
            rsp.u.data.pointer = scan_rsp->pptrs[0];
            rsp.u.data.length = scan_rsp->sizes[0];
            rsp.u.data.flags = 0;

            //chk = abcdk_socket_ioctl(SIOCGIWSCAN,&rsp);
            chk = ioctl(sock,SIOCGIWSCAN,&rsp);
            if (chk !=0)
            {
                if(errno == EAGAIN)
                    continue;
                else 
                    goto END;
            }

            abcdk_hexdump_option_t opt = {0};
            if(rsp.u.data.length)
            abcdk_hexdump(stderr,rsp.u.data.pointer,rsp.u.data.length,0,&opt);

            void *p = rsp.u.data.pointer;

            for (;p - rsp.u.data.pointer < rsp.u.data.length;)
            {
                struct iw_event *event = ABCDK_PTR2PTR(struct iw_event, p, 0);

                printf("cmd = %04X,len = %hu\n", event->cmd, event->len);

                switch (event->cmd)
                {
                case SIOCGIWAP:
                {
                    struct ether_addr *eth = ABCDK_PTR2PTR(struct ether_addr, event->u.addr.sa_data, 0);

                    printf("address: %02X:%02X:%02X:%02X:%02X:%02X\n",
                           eth->ether_addr_octet[0], eth->ether_addr_octet[1],
                           eth->ether_addr_octet[2], eth->ether_addr_octet[3],
                           eth->ether_addr_octet[4], eth->ether_addr_octet[5]);
                }
                break;
                case SIOCGIWNWID:
                {
                    if (event->u.nwid.disabled)
                        printf("\tNWID: off/any\n");
                    else
                        printf("                    NWID: %X\n", event->u.nwid.value);
                }
                break;
                case SIOCGIWFREQ:
                {
                    printf("\tchannel: %f\n",((double) event->u.freq.m) * pow(10,event->u.freq.e));
                }
                break;
                case SIOCGIWESSID:
                {
                    break;
                    event->u.essid.pointer = p+4+sizeof(struct iw_point);
                    event->u.essid.length = event->len-4-sizeof(struct iw_point);

                    char essid[4 * IW_ESSID_MAX_SIZE + 1];
                    memset(essid, '\0', sizeof(essid));
                    if ((event->u.essid.pointer) && (event->u.essid.length))
                        iw_essid_escape(essid,event->u.essid.pointer, event->u.essid.length);

                    if (event->u.essid.flags)
                    {
                        if ((event->u.essid.flags & IW_ENCODE_INDEX) > 1)
                            printf("\tESSID: %s [%d]\n",essid,(event->u.essid.flags & IW_ENCODE_INDEX));
                        else 
                            printf("\tESSID: %s\n",essid);
                    }
                    else
                    {
                        printf("\tESSID: off/any/hidden\n");
                    }
                }
                break;
                default:
                    break;
                }

                p = p+event->len;
            }

            goto END;
        

    }

END:

    abcdk_object_unref(&scan_rsp);
    abcdk_object_unref(&k);
    abcdk_object_unref(&p);

    abcdk_closep(&sock);

#else

#ifdef HAVE_LIBNM
    GError *err = NULL;

    NMClient *cli =  nm_client_new(NULL,&err);

    // NMDevice *dev = nm_client_get_device_by_iface(cli,"wlx70f11c3c3500");

    // gboolean chk = nm_device_wifi_request_scan(NM_DEVICE_WIFI(dev),NULL,&err);

    // nm_device_wifi_request_scan_async (NM_DEVICE_WIFI (dev),
	// 	                                   NULL, request_rescan_cb, cli);

    // //nm_device_wifi_request_scan_finish(&device,&cancellable,&err);

    // g_error_free(err);

#if 0

    NMDevice **devices =  get_devices_sorted (cli);

    for (int i = 0; devices[i]; i++)
    {
        NMDevice *dev = devices[i];

        if (!NM_IS_DEVICE_WIFI (dev))
            continue;

        NMAccessPoint * ap = nm_device_wifi_get_active_access_point(NM_DEVICE_WIFI (dev));
        const char * ssid = ap? nm_access_point_get_bssid (ap):"";

        printf("ssid: %s\n",ssid);
    }
#else 

    const GPtrArray *devs = nm_client_get_devices (cli);

    for(int i = 0;i<devs->len;i++)
	{
        NMDevice *dev = ((NMDevice **)devs->pdata)[i];
        printf("iface: %s\n",nm_device_get_iface(dev));
        printf("ip_iface: %s\n",nm_device_get_ip_iface(dev));
        printf("udi: %s\n",nm_device_get_udi(dev));
        printf("hw: %s\n",nm_device_get_hw_address(dev));
    }

#endif //

#endif //HAVE_LIBNM
#endif
}

void test_hexdump(abcdk_tree_t *args)
{
    const char *file_p = abcdk_option_get(args,"--file",0,"");

    abcdk_object_t * m = abcdk_mmap2(file_p,0,0);

    abcdk_hexdump_option_t opt = {0};

    if(abcdk_option_exist(args,"--show-addr"))
        opt.flag |= ABCDK_HEXDEMP_SHOW_ADDR;
    if(abcdk_option_exist(args,"--show-char"))
        opt.flag |= ABCDK_HEXDEMP_SHOW_CHAR;

    opt.width = abcdk_option_get_int(args,"--width",0,16);

    opt.keyword = abcdk_object_alloc(NULL,4,0);
    opt.palette = abcdk_object_alloc(NULL,3,0);

    opt.keyword->pptrs[0] = "mvhd";
    opt.keyword->sizes[0] = 4;
    opt.keyword->pptrs[1] = "ftyp";
    opt.keyword->sizes[1] = 4;
    opt.keyword->pptrs[2] = "moov";
    opt.keyword->sizes[2] = 4;
    opt.keyword->pptrs[3] = "mdat";
    opt.keyword->sizes[3] = 4;

    opt.palette->pptrs[0] = ABCDK_ANSI_COLOR_RED;
    opt.palette->pptrs[1] = ABCDK_ANSI_COLOR_GREEN;
    opt.palette->pptrs[2] = ABCDK_ANSI_COLOR_BLUE;

    if(m)
    {
        //ssize_t w = abcdk_hexdump(stdout,m->pptrs[0],m->sizes[0],0,&opt);
        ssize_t w = abcdk_hexdump(stdout,m->pptrs[0],1000,0,&opt);
        fprintf(stderr,"w=%ld",w);
    }

    abcdk_object_unref(&m);
    abcdk_object_unref(&opt.keyword);
    abcdk_object_unref(&opt.palette);
}

void test_video(abcdk_tree_t *args)
{
#ifdef HAVE_FFMPEG
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

    int chk;
    const char *src_file_p = abcdk_option_get(args,"--src",0,"");
    const char *dst_file_p = abcdk_option_get(args,"--dst",0,"");

    AVDictionary *dict = NULL;
#if 1
    av_dict_set(&dict,"framerate","120",0);
    //av_dict_set(&dict,"video_size","1920x1080",0);
    av_dict_set(&dict,"video_size","640x480",0);
    //av_dict_set(&dict,"input_format","mjpeg",0);
    av_dict_set(&dict,"input_format","yuyv422",0);
#endif 
    abcdk_video_t *src = abcdk_video_open_capture(NULL,src_file_p,-1UL,1,dict);

    av_dict_free(&dict);

    //abcdk_avformat_show_options(src->ctx);

    //int dst = abcdk_open(dst_file_p,1,0,1);
    abcdk_video_t *dst = abcdk_video_open_writer(NULL,dst_file_p,NULL);

    int stream_index = abcdk_video_find_stream(src,1);

    double fps = abcdk_video_get_fps(src,stream_index);
    int width = abcdk_video_get_width(src,stream_index);
    int height = abcdk_video_get_height(src,stream_index);
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,35,100) 
    enum AVCodecID id = src->ctx->streams[stream_index]->codec->codec_id;
#else 
    enum AVCodecID id = src->ctx->streams[stream_index]->codecpar->codec_id;
#endif 

    //int stream_index2 = abcdk_video_add_stream(dst,fps,width,height,id,NULL,0,0);
    int stream_index2 = abcdk_video_add_stream(dst,fps,width,height,AV_CODEC_ID_H264,NULL,0,0);

    //  int stream_index2 = abcdk_video_add_stream(dst, fps, width, height, id,
    //                                             src->ctx->streams[stream_index]->codec->extradata,
    //                                             src->ctx->streams[stream_index]->codec->extradata_size,
    //                                             1);

    uint64_t c = 0;
    uint64_t s = 0;

    abcdk_video_write_header(dst,0,1);

    printf("LONG: %f\n",abcdk_video_get_duration(src,stream_index));
    printf("FPS: %f\n",abcdk_video_get_fps(src,stream_index));

    AVPacket pkt;
    av_init_packet(&pkt);
    AVFrame *fae = av_frame_alloc();

    AVFrame *fae2 = av_frame_alloc();
    fae2->format = dst->codec_ctx[0]->pix_fmt;
    fae2->height = height;
    fae2->width = width;
    av_frame_get_buffer(fae2,1);

    struct SwsContext *sws = NULL;

    for(int i =0;i<1000;i++)
    {   
      //  chk = abcdk_video_read(src,&pkt,stream_index,0,1);
        chk = abcdk_video_read2(src,fae,stream_index,0);
        if(chk < 0)
            break;

        printf("DTS: %f ,PTS: %f\n",
          //     abcdk_video_ts2sec(src, pkt.stream_index, pkt.dts),
         //     abcdk_video_ts2sec(src, pkt.stream_index, pkt.pts));
              abcdk_video_ts2sec(src, chk, fae->pkt_dts),
               abcdk_video_ts2sec(src, chk, fae->pkt_pts));

        // abcdk_write(dst,pkt.data,pkt.size);

        // chk = abcdk_video_write3(dst,stream_index2,pkt.data,pkt.size);

        if(!sws)
            sws = abcdk_sws_alloc2(fae, fae2, 0);

        abcdk_sws_scale(sws,fae,fae2);

         chk = abcdk_video_write2(dst,stream_index2,fae2);
         if(chk < 0)
            break;

         s = abcdk_clock(c, &c) / 1000;
         if (s < (1000 / fps))
             usleep(((1000 / fps) - s) * 1000);
    }
    av_frame_free(&fae);
    av_packet_unref(&pkt);

    abcdk_video_write_trailer(dst);

   // abcdk_closep(&dst);
    abcdk_video_close(dst);
    abcdk_video_close(src);
#pragma GCC diagnostic pop
#endif //
}


void test_com(abcdk_tree_t *args)
{
    const char *port = abcdk_option_get(args,"--port",0,"");

    int fd = open(port,O_RDWR|O_NOCTTY);


 //   assert(isatty(fd)==0);

#if 0
    struct termios opt = {0};

    int chk = tcgetattr(fd,&opt);
    
   // tcflush(fd, TCIOFLUSH);
    cfsetispeed(&opt,B9600);
    cfsetospeed(&opt,B9600);
    
    opt.c_cflag |=(CLOCAL|CREAD);
    opt.c_cflag &= ~PARENB; 
    opt.c_cflag &= ~CSTOPB; 
    opt.c_cflag &= ~CSIZE; 
    opt.c_cflag |= ~CS8;
    opt.c_cc[VTIME] = 0;
    opt.c_cc[VMIN] = 0;

    tcflush(fd,TCIOFLUSH);  

    //cfsetispeed(&opt,B4800);
    assert(tcsetattr(fd,TCSANOW,&opt)==0);



    struct serial_rs485 conf = {0};

    conf.flags |= SER_RS485_ENABLED;
   // conf.flags |= SER_RS485_RX_DURING_TX;

   // assert(ioctl(fd,TIOCSRS485,&conf)==0);
#else 
    assert(abcdk_tcattr_serial(fd, 9600, 8, 0, 1,NULL)== 0);
#endif 
    uint64_t s = 0,s1 = 0,s2 = 0;
    char buf1[18]={0};
    char buf2[18]={0};
    for(int i = 0;i<999999999;i++)
    {

        int chk = abcdk_poll(fd,0x01,-1);
        assert(chk>0);

        abcdk_read(fd,buf1,17);

        s1 = abcdk_clock(s,&s);
        s2 += s1;

        if(memcmp(buf1,buf2,17)!=0)
            s2 = 0;
        else if(s2 >= 1000000)
            s2 = 0;

        if(s2 ==0 )
        {
            memcpy(buf2,buf1,17);

            char buf3[35]={0};
            abcdk_bin2hex(buf3,buf1,17,0);

            printf("[%d]: '%s' '%s'\n",i,buf1,buf3);
        }

    }


    abcdk_closep(&fd);
}

void test_mpi(abcdk_tree_t *args)
{

#ifdef  HAVE_MPI
    // int argc = 1;
    // char *argv[1] = {
    //     abcdk_option_get(args,"--",0,""),
    // };

    int rank,size;

 //   MPI_Init(&argc, &argv);
    MPI_Init(NULL,NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello World from thread %d of %d\n", rank, size);

    MPI_Finalize();

#endif 
}

void test_lz4(abcdk_tree_t *args)
{
#ifdef  HAVE_LZ4

    const char *src = abcdk_option_get(args,"--src",0,"");
    const char *dst = abcdk_option_get(args,"--dst",0,"");

    abcdk_object_t *s = abcdk_mmap2(src,0,0);

    size_t dsize = abcdk_endian_b_to_h32(ABCDK_PTR2U32(s->pptrs[0],0));

    abcdk_object_t *d = abcdk_object_alloc2(dsize);

    //LZ4_decompress_fast(s->pptrs[0]+4,d->pptrs[0],dsize);
    int m = abcdk_lz4_dec_fast(d->pptrs[0],dsize,s->pptrs[0]+4);

    abcdk_object_t *q = abcdk_object_alloc2(2000);

    int n = abcdk_lz4_enc_default(q->pptrs[0],q->sizes[0],d->pptrs[0],dsize);

    //assert(memcmp(q->pptrs[0],s->pptrs[0]+4,s->sizes[0]-4)==0);

    abcdk_object_t *p = abcdk_object_alloc2(dsize);

    int m2 = abcdk_lz4_dec_fast(p->pptrs[0],dsize,q->pptrs[0]);

    assert(memcmp(p->pptrs[0],d->pptrs[0],d->sizes[0])==0);

    abcdk_object_unref(&q);
    abcdk_object_unref(&p);

    int fd = abcdk_open(dst,1,0,1);
    ftruncate(fd,0);
    abcdk_write(fd,d->pptrs[0],dsize);
    abcdk_closep(&fd);

    abcdk_object_unref(&s);
    abcdk_object_unref(&d);


#endif 
}

#ifdef HAVE_ARCHIVE

static struct _test_archive_store
{
    int fd;
    const char *volume;
} test_archive_store[] = {
    {-1,"/home/devel/remote/192.167.15.189-mnt/zhangpengcheng/bbbb.tar"},
//    {-1,"/home/devel/remote/192.167.15.190-mnt/zhangpengcheng/bbbb.tar"},
//    {-1,"/home/devel/remote/192.167.15.188-mnt/zhangpengcheng/bbbb.tar"}
  //  {-1,"/home/devel/job/tmp/bbbb.tar"},
    {-1,"/tmp/bbbb.tar"}
   //   {-1,"/dev/nst0"}
};

ssize_t test_archive_write_cb(struct archive *fd, void *_client_data, const void *_buffer, size_t _length)
{
    int num = ABCDK_ARRAY_SIZE(test_archive_store);
    ssize_t wlen = 0,wall = 0;

#pragma omp parallel for num_threads(num)
    for (int i = 0; i < num; i++)
    {
        wlen = abcdk_write(test_archive_store[i].fd, _buffer, _length);
#pragma omp atomic
        wall += ((wlen>0)?wlen:0);
    }

    return wall/num;
}

int test_archive_open_cb(struct archive *fd, void *_client_data)
{
    int num = ABCDK_ARRAY_SIZE(test_archive_store);

    for (int i = 0; i < num; i++)
    {
        test_archive_store[i].fd = abcdk_open(test_archive_store[i].volume, 1, 0, 1);
    }

    return ARCHIVE_OK;
}

int test_archive_close_cb(struct archive *fd, void *_client_data)
{
    int num = ABCDK_ARRAY_SIZE(test_archive_store);

    for (int i = 0; i < num; i++)
    {
        abcdk_closep(&test_archive_store[i].fd);
    }

    return ARCHIVE_OK;
}

#endif

void test_archive(abcdk_tree_t *args)
{
#ifdef HAVE_ARCHIVE

    const char *src = abcdk_option_get(args,"--src",0,"");
    const char *dst = abcdk_option_get(args,"--dst",0,"");

    struct archive *a = archive_write_new();


    archive_write_set_bytes_per_block(a,256*1024);

    

  //  archive_write_add_filter_bzip2(a);
  //  archive_write_set_format_zip(a);

  //  archive_write_add_filter_gzip(a);
  //  archive_write_set_format_pax_restricted(a); // Note 1

    archive_write_set_format_gnutar(a);

  //  archive_write_open_filename(a, dst);
    archive_write_open(a,NULL,test_archive_open_cb,test_archive_write_cb,test_archive_close_cb);


    struct archive_entry *entry = archive_entry_new();

    int fd = abcdk_open(src,0,0,0);

    struct stat st = {0};
    fstat(fd,&st);

    archive_entry_copy_pathname(entry,src+10);



#if 0
    archive_entry_set_size(entry, st.st_size); // Note 3
    archive_entry_set_filetype(entry, AE_IFREG);
    archive_entry_set_perm(entry, 0644);
#else 
    archive_entry_copy_stat(entry,&st);
#endif 

    

    archive_write_header(a, entry);

    int bufsize = 256*1024;
    char *buf = (char*)abcdk_heap_alloc(bufsize);

    for(;;)
    {
        ssize_t r = abcdk_read(fd,buf,bufsize);
        if(r<=0)
            break;
        
        archive_write_data(a,buf,r);
    }

    abcdk_heap_free(buf);

    archive_write_finish_entry(a);
    archive_entry_free(entry);

    abcdk_closep(&fd);

    archive_write_close(a);
	archive_write_free(a);

    
    
#endif
}

void test_modbus(abcdk_tree_t *args)
{
#ifdef HAVE_MODBUS
    const char *port = abcdk_option_get(args,"--port",0,"");

    modbus_t *m = modbus_new_rtu(port, 9600, 'N', 8, 1);
    modbus_set_debug(m, 0);
    modbus_set_slave(m,1);
    modbus_connect(m);

    struct timeval t;
    t.tv_sec = 10;
    t.tv_usec = 0;
    modbus_set_response_timeout(m, &t);

    //int chk = modbus_rtu_set_serial_mode(m,MODBUS_RTU_RS232);

    int f2 = 0;
    while(1)
    {
        uint16_t buf[20]={0};
        int regs = modbus_read_registers(m,3,2,buf);

        int f = ABCDK_PTR2OBJ(float, buf, 0) * 1000;
        if (f != f2)
        {
            printf("%f\n", (float)f / 1000);
            f2 = f;
        }

        usleep(1000);
    }

  
    modbus_close(m);
    modbus_free(m);
  

#endif
}
#ifdef HAVE_LIBUSB

static void print_devs(libusb_device **devs)
{
	libusb_device *dev;
	int i = 0, j = 0;
	uint8_t path[8]; 

	while ((dev = devs[i++]) != NULL) {
		struct libusb_device_descriptor desc;
		int r = libusb_get_device_descriptor(dev, &desc);
		if (r < 0) {
			fprintf(stderr, "failed to get device descriptor");
			return;
		}

		printf("%04x:%04x (bus %d, device %d)",
			desc.idVendor, desc.idProduct,
			libusb_get_bus_number(dev), libusb_get_device_address(dev));

		r = libusb_get_port_numbers(dev, path, sizeof(path));
		if (r > 0) {
			printf(" path: %d", path[0]);
			for (j = 1; j < r; j++)
				printf(".%d", path[j]);
		}
		printf("\n");
	}
}
#endif

int test_libusb(abcdk_tree_t *args)
{
#ifdef HAVE_LIBUSB
	libusb_device **devs;
	int r;
	ssize_t cnt;

	r = libusb_init(NULL);
	if (r < 0)
		return r;

	cnt = libusb_get_device_list(NULL, &devs);
	if (cnt < 0){
		libusb_exit(NULL);
		return (int) cnt;
	}

	print_devs(devs);
	libusb_free_device_list(devs, 1);

	libusb_exit(NULL);
#endif
    return 0;
}

#ifdef HAVE_OPENSSL

void test_openssl_server(abcdk_tree_t *args)
{
#if OPENSSL_VERSION_NUMBER <= 0x100020bfL  
    const SSL_METHOD *method = TLSv1_2_server_method();
#else
    const SSL_METHOD *method = TLS_server_method();
#endif 

    SSL_CTX * ctx = SSL_CTX_new(method);
    int chk;
    const char *capath = abcdk_option_get(args,"--ca-path",0,NULL);

    if (capath)
    {
        /*如果使用证书路径加载证书，则需要使用工具生成证收的hash文件名。c_rehash <CApath> */
        chk = SSL_CTX_load_verify_locations(ctx, NULL, capath);
        assert(chk == 1);

        X509_VERIFY_PARAM *param = SSL_CTX_get0_param(ctx);
        //X509_VERIFY_PARAM_set_purpose(param, X509_PURPOSE_ANY);
        X509_VERIFY_PARAM_set_flags(param, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);
    }

    chk = abcdk_openssl_ssl_ctx_load_crt(ctx, abcdk_option_get(args, "--crt-file", 0, NULL),
                                          abcdk_option_get(args, "--key-file", 0, NULL),
                                          abcdk_option_get(args, "--key-pwd", 0, NULL));

    SSL* s = abcdk_openssl_ssl_alloc(ctx);


     SSL_set_verify(s,SSL_VERIFY_PEER,NULL);


    abcdk_sockaddr_t addr = {0};
    //abcdk_sockaddr_from_string(&addr,"0.0.0.0:12345",0);
    addr.family = AF_UNIX;
    strcpy(addr.addr_un.sun_path,"/tmp/abcdk.txt2");

    int l = abcdk_socket(addr.family,0);

    int flag = 1;
    abcdk_sockopt_option_int(l, SOL_SOCKET, SO_REUSEPORT, &flag, 2);
    abcdk_sockopt_option_int(l, SOL_SOCKET, SO_REUSEADDR, &flag, 2);

    unlink(addr.addr_un.sun_path);

    assert(abcdk_bind(l,&addr)==0);
    assert(listen(l, SOMAXCONN)==0);

    abcdk_sockaddr_t addr2 = {0};
    int c = abcdk_accept(l,&addr2);

    assert(abcdk_openssl_ssl_handshake(c,s,1,10000)==0);

    int chk2 = SSL_get_verify_result(s);
    printf("chk2 = %d\n",chk2);
    //assert(X509_V_OK == chk2);
    
    char buf[100]={0};
    SSL_read(s,buf,5);

    printf("{%s}\n",buf);

    SSL_write(s,"abcdk",5);


    abcdk_closep(&c);
    abcdk_closep(&l);

    abcdk_openssl_ssl_free(&s);

    SSL_CTX_free(ctx);
}

void test_openssl_client(abcdk_tree_t *args)
{
#if OPENSSL_VERSION_NUMBER <= 0x100020bfL  
    const SSL_METHOD *method = TLSv1_2_client_method();
#else
    const SSL_METHOD *method = TLS_client_method();
#endif 
    int chk ;
    SSL_CTX * ctx = SSL_CTX_new(method);

    const char *capath = abcdk_option_get(args,"--ca-path",0,NULL);

    if (capath)
    {
        chk = SSL_CTX_load_verify_locations(ctx, NULL, capath);
        assert(chk == 1);

        X509_VERIFY_PARAM *param = SSL_CTX_get0_param(ctx);
        //X509_VERIFY_PARAM_set_purpose(param, X509_PURPOSE_ANY);
        X509_VERIFY_PARAM_set_flags(param, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);
    }

    chk = abcdk_openssl_ssl_ctx_load_crt(ctx, abcdk_option_get(args, "--crt-file", 0, NULL),
                                          abcdk_option_get(args, "--key-file", 0, NULL),
                                          abcdk_option_get(args, "--key-pwd", 0, NULL));

    assert(chk == 0);

    SSL* s = abcdk_openssl_ssl_alloc(ctx);

    // void *p = SSL_get_app_data(s);
    // printf("p = %p\n",p);


    SSL_set_verify(s,SSL_VERIFY_PEER,NULL);

    abcdk_sockaddr_t addr = {0};
  //  abcdk_sockaddr_from_string(&addr,
  //                             abcdk_option_get(args, "--server-addr", 0, "localhost:12345"),
  //                             1);

    addr.family = AF_UNIX;
    strcpy(addr.addr_un.sun_path,"/tmp/abcdk.txt2");

    int c = abcdk_socket(addr.family,0);
    
    assert(abcdk_connect(c,&addr,10000)==0);

    assert(abcdk_openssl_ssl_handshake(c,s,0,10000)==0);

    int chk2 = SSL_get_verify_result(s);
    printf("chk2 = %d\n",chk2);
    //assert(X509_V_OK == chk2);

    SSL_write(s,"abcdk",5);

    char buf[100]={0};
    SSL_read(s,buf,100);

    printf("{%s}\n",buf);

    abcdk_closep(&c);

    abcdk_openssl_ssl_free(&s);

    SSL_CTX_free(ctx);
}

#endif

int test_openssl(abcdk_tree_t *args)
{
    int sub_func = abcdk_option_get_int(args, "--sub-func", 0, 0);

#ifdef HAVE_OPENSSL

    if (sub_func == 1)
        test_openssl_server(args);
    else if (sub_func == 2)
        test_openssl_client(args);

#endif

    return 0;
}

#ifdef HAVE_MQTT
void my_message_callback(struct mosquitto *mosq, void *userdata, const struct mosquitto_message *message)
{
    if (message->payloadlen)
    {
#ifdef _json_h_
        json_object *obj = abcdk_json_parse((char*)message->payload);

        //abcdk_json_readable(stderr,1,10,obj);
        //fprintf(stderr,"\n");

        json_object *mid = NULL,*node = NULL,*rt = NULL,*cmd = NULL;
        json_object_object_get_ex(obj,"mid",&mid);
        json_object_object_get_ex(obj,"node",&node);
        json_object_object_get_ex(obj,"realtime",&rt);
        json_object_object_get_ex(obj,"cmd",&cmd);

        int64_t brt = json_object_get_int64(rt);
        int64_t ert = abcdk_time_clock2kind_with(CLOCK_REALTIME,3);
        

        if(strncmp("543",json_object_get_string(node),3)==0)
            {
                syslog(LOG_DEBUG,"mid:%s,node:%s,cmd=%d,q=%ld",
                        json_object_get_string(mid),
                        json_object_get_string(node),
                        json_object_get_int(cmd),
                        ert-brt);
            }

        
        abcdk_json_unref(&rt);
        abcdk_json_unref(&mid);
        abcdk_json_unref(&node);
        abcdk_json_unref(&obj);

#else 
        printf("%s %s\n", message->topic, (char*)message->payload);
#endif 
        
    }
    else
    {
        printf("%s (null)\n", message->topic);
    }
    fflush(stdout);
}

void my_connect_callback(struct mosquitto *mosq, void *userdata, int result)
{
    int i;
    if (!result)
    {
        /* Subscribe to broker information topics on successful connect. */
        //mosquitto_subscribe(mosq, NULL, "$SYS/#", 2);
        //mosquitto_subscribe(mosq, NULL, "hello", 2);
        mosquitto_subscribe(mosq, NULL, "dagger/manager-node", 0);
    }
    else
    {
        fprintf(stderr, "Connect failed\n");
    }
}

void my_subscribe_callback(struct mosquitto *mosq, void *userdata, int mid, int qos_count, const int *granted_qos)
{
    int i;
#if 0
    printf("Subscribed (mid: %d): %d", mid, granted_qos[0]);
    for (i = 1; i < qos_count; i++)
    {
        printf(", %d", granted_qos[i]);
    }
    printf("\n");
#endif
}

void my_log_callback(struct mosquitto *mosq, void *userdata, int level, const char *str)
{
    /* Pring all log messages regardless of level. */
#if 0
    syslog(LOG_DEBUG,"%s\n", str);
#endif
}
#endif 

int test_mqtt(abcdk_tree_t *args)
{
#ifdef HAVE_MQTT
    int i;
    char *host = "192.167.200.102";
    int port = 1883;
    int keepalive = 60;
    bool clean_session = true;
    struct mosquitto *mosq = NULL;

    mosquitto_lib_init();
    mosq = mosquitto_new(NULL, clean_session, NULL);
    if (!mosq)
    {
        fprintf(stderr, "Error: Out of memory.\n");
        return 1;
    }
    mosquitto_log_callback_set(mosq, my_log_callback);
    mosquitto_connect_callback_set(mosq, my_connect_callback);
    mosquitto_message_callback_set(mosq, my_message_callback);
    mosquitto_subscribe_callback_set(mosq, my_subscribe_callback);

    if (mosquitto_connect(mosq, host, port, keepalive))
    {
        fprintf(stderr, "Unable to connect.\n");
        return 1;
    }

    mosquitto_loop_forever(mosq, -1, 1);

    

    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();

#endif 

    return 0;
}

void test_http(abcdk_tree_t *args)
{
    int s = abcdk_socket(AF_INET,0);

    abcdk_sockaddr_t a;
    a.family = AF_INET;
    abcdk_sockaddr_from_string(&a,"0.0.0.0:12345",0);
    abcdk_bind(s,&a);
    listen(s,10);

    int c = abcdk_accept(s,NULL);

    char buf[10]={0};

    printf("--->>>\r\n");

    while(read(c,buf,1)>0)
    {
        printf("%s",buf);
    }

    printf("<<<---\r\n");

    abcdk_closep(&c);
    abcdk_closep(&s);
    
}

void test_redis(abcdk_tree_t *args)
{
#ifdef __HIREDIS_H

    const char *server = abcdk_option_get(args, "--server", 0, "127.0.0.1");
    int port = abcdk_option_get_int(args, "--port", 0, 6379);

    redisContext *c = abcdk_redis_connect(server, port, 20);
    if (!c)
        return;

    //printf("%s\n", c->errstr);

    int chk = abcdk_redis_auth(c,"12345678");
    assert(chk==0);

    char buf[128]={0};
    abcdk_redis_get_auth(c,buf);
    printf("{%s}\n",buf);

    chk = abcdk_redis_set_auth(c,"12345678");
    assert(chk==0);

    chk = abcdk_redis_auth(c,"12345678");
    assert(chk==0);

    char buf2[128]={0};
    abcdk_redis_get_auth(c,buf2);
    printf("{%s}\n",buf2);

    redisFree(c);
#endif //
}


void test_cert_verify(abcdk_tree_t *args)
{
#ifdef HAVE_OPENSSL

    const char *user = abcdk_option_get(args, "--user-crt", 0, "");

    //SSLeay_add_all_algorithms();

    X509 *cert = abcdk_openssl_load_crt(user,NULL);

    //PEM_read_X509_CRL()
 
    X509_STORE *store = X509_STORE_new();

    for(int i = 0;i<100;i++)
    {
        const char *ca = abcdk_option_get(args,"--ca-crt",i,NULL);
        if(!ca)
            break;

        abcdk_openssl_load_crt2store(store,ca,NULL);
    }


    for(int i = 0;i<100;i++)
    {
        const char *ca = abcdk_option_get(args,"--ca-crl",i,NULL);
        if(!ca)
            break;

        abcdk_openssl_load_crl2store(store,ca,NULL);
    }

    

    X509_STORE_CTX *store_ctx = abcdk_openssl_verify_crt_prepare(store,cert);

    X509_VERIFY_PARAM *param = X509_STORE_CTX_get0_param(store_ctx);

   // X509_VERIFY_PARAM_set_purpose(param, X509_PURPOSE_ANY);
    /*
     * X509_V_FLAG_CRL_CHECK 只验证叶证书是否被吊销，并且只要求叶证书的父级证书存在吊销列表即可。
     * X509_V_FLAG_CRL_CHECK_ALL 验证证书链，并且要求所有父级证书(根证书除外)的吊销列表都存在。
     * 
     * X509_V_FLAG_CRL_CHECK_ALL 单独启用无效，至少要配合X509_V_FLAG_CRL_CHECK启用。
    */
    X509_VERIFY_PARAM_set_flags(param,X509_V_FLAG_CRL_CHECK|X509_V_FLAG_CRL_CHECK_ALL);
  //  X509_VERIFY_PARAM_set_flags(param,X509_V_FLAG_CRL_CHECK);

    int chk = X509_verify_cert(store_ctx);
    assert(chk == 1);

    //X509_VERIFY_PARAM_free(param);

    X509_free(cert);
    X509_STORE_free(store);
    X509_STORE_CTX_free(store_ctx);

#endif 
}

void test_json(abcdk_tree_t *args)
{
#ifdef _json_h_

    const char *src = abcdk_option_get(args,"--src",0,NULL);

    json_object *src_obj = json_object_from_file(src);

    abcdk_json_readable(stdout,1,0,src_obj);
    printf("\n");

    json_object *it = abcdk_json_locate(src_obj,"frame","detector",NULL);

    json_object *it2 = json_object_array_get_idx(it,0);

    json_object *it3 = abcdk_json_locate(it2,"?box",NULL);

    printf("'%s'\n",json_object_get_string(it3));

    abcdk_json_unref(&src_obj);

#endif //_json_h_
}

void test_refer_count(abcdk_tree_t *args)
{
    int user = abcdk_option_get_int(args,"--user",0,10);

    abcdk_object_t * p= abcdk_object_alloc2(100);

#pragma omp parallel for num_threads(user)
    for (int i = 0; i < 100000; i++)
    {
        abcdk_object_t *q = abcdk_object_refer(p);

        usleep(10*1000);

        abcdk_object_unref(&q);
    }

    abcdk_object_unref(&p);
}

typedef struct _one_node
{
    int id;

    abcdk_comm_message_t *in_buffer;

    abcdk_comm_message_t *out_buffer;
    abcdk_comm_queue_t *out_queue;

    abcdk_comm_node_t *node;

    abcdk_comm_waiter_t *rsp;

}one_node_t;

int smb_protocol(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
{
    size_t off = abcdk_comm_message_offset(msg);
    if (off < 4)
        return 0;

    size_t len = abcdk_endian_b_to_h32(ABCDK_PTR2U32(abcdk_comm_message_data(msg), 0));
    if (len != abcdk_comm_message_size(msg))
    {
        abcdk_comm_message_realloc(msg, len);
        return 0;
    }
    else if (len != abcdk_comm_message_offset(msg))
    {
        return 0;
    }

    return 1;
}

void _output_event(one_node_t *one)
{
    int chk;

NEXT_MSG:

    if (!one->out_buffer)
    {
        one->out_buffer = abcdk_comm_queue_pop(one->out_queue);
        if (!one->out_buffer)
            return;
    }

    chk = abcdk_comm_message_send(one->node, one->out_buffer);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(one->node, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_write_watch(one->node);
        return;
    }
    
    /*释放消息缓存，并继续发送。*/
    abcdk_comm_message_unref(&one->out_buffer);
    goto NEXT_MSG;
}

void test_comm_message_cb(abcdk_comm_node_t *node, uint32_t event)
{
    one_node_t *one = (one_node_t *)abcdk_comm_get_userdata(node);

    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
    {
        assert(one == NULL);
        one = (one_node_t*)abcdk_heap_alloc(sizeof(one_node_t));
        one->out_queue = abcdk_comm_queue_alloc();
        one->node = abcdk_comm_node_refer(node);
        abcdk_comm_set_userdata(node,one);

        abcdk_comm_read_watch(node);
    }
        break;
    case ABCDK_COMM_EVENT_INPUT:
        {
            if(!one->in_buffer)
            {
                one->in_buffer = abcdk_comm_message_alloc(4);
                abcdk_comm_message_protocol_set(one->in_buffer,smb_protocol);
            }
            
            int chk = abcdk_comm_message_recv(node,one->in_buffer);
            if(chk < 0)
            {
                abcdk_comm_set_timeout(node,1);
            }
            else if(chk == 0)
            {
                abcdk_comm_read_watch(node);
            }
            else
            {
                abcdk_comm_message_t *msg_copy = abcdk_comm_message_refer(one->in_buffer);
                abcdk_comm_message_unref(&one->in_buffer);
                abcdk_comm_read_watch(node);

            //    usleep(rand()%10000+1000);

                abcdk_comm_message_reset(msg_copy);
                abcdk_comm_queue_push(one->out_queue,msg_copy);
                abcdk_comm_write_watch(one->node);
            }
        }
        break;
    case ABCDK_COMM_EVENT_OUTPUT:
            _output_event(one);
        break;
    case ABCDK_COMM_EVENT_CLOSE:
    default:
    {
        char sockname_str[100] = {0}, peername_str[100] = {0};
        abcdk_comm_get_sockaddr_str(one->node, sockname_str,peername_str);

        printf("Socket: %s -> %s Disconnected.\n", sockname_str, peername_str);

        if(one)
        {
        abcdk_comm_message_unref(&one->in_buffer);
        abcdk_comm_queue_free(&one->out_queue);
        abcdk_comm_node_unref(&one->node);
        abcdk_heap_free(one);
        }
    }
    break;
    }
}

void *test_send_msg(void *args)
{
    one_node_t *one = (one_node_t *)args;

    for (int i = 0; i < 1000; i++)
    {
      //  usleep(10);

        abcdk_comm_message_t *msg = abcdk_comm_message_alloc(128);

        uint64_t mid = abcdk_time_clock2kind_with(0, 6);

        abcdk_comm_waiter_request2(one->rsp,&mid);

        ABCDK_PTR2U32(abcdk_comm_message_data(msg), 0) = abcdk_endian_h_to_b32(128);
        ABCDK_PTR2U64(abcdk_comm_message_data(msg), 4) = abcdk_endian_h_to_b64(mid);
        ABCDK_PTR2U32(abcdk_comm_message_data(msg), 12) = abcdk_endian_h_to_b32(i+1);

        abcdk_comm_queue_push(one->out_queue, msg);
        abcdk_comm_write_watch(one->node);

        abcdk_comm_queue_t * q = abcdk_comm_waiter_wait2(one->rsp,&mid,1,10);
        if(!q)
            continue;

        uint64_t a = abcdk_time_clock2kind_with(0,6);

        printf("mid(%lu),timeout(%lu), count(%lu)\n",mid,a-mid,abcdk_comm_queue_count(q));
        
        abcdk_comm_queue_free(&q);
    }

    return NULL;
}



void test_comm_message2_cb(abcdk_comm_node_t *node, uint32_t event)
{

    one_node_t *one = (one_node_t *)abcdk_comm_get_userdata(node);

    switch (event)
    {
    case ABCDK_COMM_EVENT_CONNECT:
        {
            one->out_queue = abcdk_comm_queue_alloc();
            one->rsp = abcdk_comm_waiter_alloc();
            one->node = abcdk_comm_node_refer(node);
        //    abcdk_comm_set_userdata(node,one);

            abcdk_comm_read_watch(node);

            abcdk_thread_t t;
            t.routine = test_send_msg;
            t.opaque = one;

            abcdk_thread_create(&t,0);
            
        }
        break;
    case ABCDK_COMM_EVENT_INPUT:
        {
            if(!one->in_buffer)
            {
                one->in_buffer = abcdk_comm_message_alloc(4);
                abcdk_comm_message_protocol_set(one->in_buffer,smb_protocol);
            }

            int chk = abcdk_comm_message_recv(node,one->in_buffer);
            if(chk != 1)
            {
                abcdk_comm_read_watch(node);
            }
            else
            {
                abcdk_comm_message_t *msg_copy = abcdk_comm_message_refer(one->in_buffer);
                abcdk_comm_message_unref(&one->in_buffer);

                abcdk_comm_read_watch(node);

                size_t len = abcdk_endian_b_to_h32(ABCDK_PTR2U32(abcdk_comm_message_data(msg_copy),0));
                uint64_t mid = abcdk_endian_b_to_h64(ABCDK_PTR2U64(abcdk_comm_message_data(msg_copy),4));
                uint32_t id = abcdk_endian_b_to_h32(ABCDK_PTR2U32(abcdk_comm_message_data(msg_copy), 12));
                uint64_t a = abcdk_time_clock2kind_with(0,3);

                //printf("mid=%lu,id=%u,time=%lu\n",mid,id,a-mid);

                abcdk_comm_waiter_response2(one->rsp,&mid,msg_copy);

               // abcdk_comm_message_unref(&msg_copy);
                
            }
        }
        break;
    case ABCDK_COMM_EVENT_OUTPUT:
           _output_event(one);
        break;
    default:
    {
        char sockname_str[100] = {0}, peername_str[100] = {0};
        abcdk_comm_get_sockaddr_str(one->node, sockname_str, peername_str);

        printf("Socket: %s -> %s Disconnected.\n", sockname_str, peername_str);

        abcdk_comm_message_unref(&one->in_buffer);
        abcdk_comm_queue_free(&one->out_queue);
        abcdk_comm_node_unref(&one->node);
        abcdk_comm_waiter_free(&one->rsp);
        abcdk_heap_free(one);
    }
    break;
    }
}


void test_comm(abcdk_tree_t *args)
{
    signal(SIGPIPE,NULL);

    abcdk_comm_t *ctx = abcdk_comm_start(0);

    SSL_CTX *server_ssl_ctx = NULL;
    SSL_CTX *client_ssl_ctx = NULL;

#ifdef HAVE_OPENSSL

    const char *capath = abcdk_option_get(args,"--ca-path",0,NULL);

    if (capath)
    {
        server_ssl_ctx = abcdk_openssl_ssl_ctx_alloc(1, NULL, capath, 2);

        abcdk_openssl_ssl_ctx_load_crt(server_ssl_ctx, abcdk_option_get(args, "--crt-file", 0, NULL),
                                       abcdk_option_get(args, "--key-file", 0, NULL),
                                       abcdk_option_get(args, "--key-pwd", 0, NULL));

   //     SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);

    //    SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER, NULL);

        client_ssl_ctx = abcdk_openssl_ssl_ctx_alloc(0, NULL, capath, 2);

        abcdk_openssl_ssl_ctx_load_crt(client_ssl_ctx, abcdk_option_get(args, "--crt2-file", 0, NULL),
                                       abcdk_option_get(args, "--key2-file", 0, NULL),
                                       abcdk_option_get(args, "--key2-pwd", 0, NULL));

   //     SSL_CTX_set_verify(client_ssl_ctx, SSL_VERIFY_PEER, NULL);

    }
#endif //HAVE_OPENSSL

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_t addr2 = {0};

    const char *listen_p = abcdk_option_get(args,"--listen",0,"0.0.0.0:12345");
    abcdk_sockaddr_from_string(&addr,listen_p,0);

    abcdk_comm_listen(ctx,server_ssl_ctx,&addr,test_comm_message_cb,NULL);

    const char *connect_p = abcdk_option_get(args,"--connect",0,"127.0.0.1:12345");
    abcdk_sockaddr_from_string(&addr2,connect_p,0);
    abcdk_comm_connect(ctx,client_ssl_ctx,&addr2,test_comm_message2_cb,abcdk_heap_alloc(sizeof(one_node_t)));
 


    while (getchar() != 'Q')
        ;

    abcdk_comm_stop(&ctx);

}

void test_easy_request_cb(abcdk_comm_easy_t *easy, const void *data, size_t len)
{
    char sockname_str[NAME_MAX] = {0}, peername_str[NAME_MAX] = {0};

    abcdk_comm_easy_get_sockaddr_str(easy,sockname_str,peername_str);

 //   printf("Server(%s -> %s): \n", sockname_str, peername_str);

    if(!data)
    {
        printf(" Disconnected.\n");
    }
    else
    {
            uint64_t a = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6);
            uint64_t b = atoll((char*)data);

    //       printf("%lu-%lu=%lu\n",a,b,a-b);

    //    usleep(rand()%10000+1000);

        abcdk_comm_easy_response(easy,data,len);
        abcdk_comm_easy_request(easy,data,len,NULL);


    }
}

void test_easy_request2_cb(abcdk_comm_easy_t *easy, const void *data, size_t len)
{
    char sockname_str[NAME_MAX] = {0}, peername_str[NAME_MAX] = {0};
    
    abcdk_comm_easy_get_sockaddr_str(easy,sockname_str,peername_str);

   // printf("Client(%s -> %s): \n", sockname_str, peername_str);

    if(!data)
    {
        printf(" Disconnected.\n");
    }
    else
    {
     //   printf(" %s\n",(char*)data);
    }
}


void test_easy(abcdk_tree_t *args)
{
    signal(SIGPIPE,NULL);

    abcdk_comm_t *ctx = abcdk_comm_start(0);

    SSL_CTX *server_ssl_ctx = NULL;
    SSL_CTX *client_ssl_ctx[4] = {NULL};

#ifdef HAVE_OPENSSL

    const char *capath = abcdk_option_get(args,"--ca-path",0,NULL);

    if (capath)
    {
        server_ssl_ctx = abcdk_openssl_ssl_ctx_alloc(1, NULL, capath, 2);

        abcdk_openssl_ssl_ctx_load_crt(server_ssl_ctx, abcdk_option_get(args, "--crt-file", 0, NULL),
                                       abcdk_option_get(args, "--key-file", 0, NULL),
                                       abcdk_option_get(args, "--key-pwd", 0, NULL));

   //     SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);

        SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER, NULL);

        for(int i =0;i<4;i++)
        {
            client_ssl_ctx[i] = abcdk_openssl_ssl_ctx_alloc(0, NULL, capath, 2);

        abcdk_openssl_ssl_ctx_load_crt(client_ssl_ctx[i], abcdk_option_get(args, "--crt2-file", i, NULL),
                                       abcdk_option_get(args, "--key2-file", i, NULL),
                                       abcdk_option_get(args, "--key2-pwd", i, NULL));

            SSL_CTX_set_verify(client_ssl_ctx[i], SSL_VERIFY_PEER, NULL);
        }

    }
#endif //HAVE_OPENSSL

    const char *sunpath = "/tmp/test_easy.sock";
    unlink(sunpath);

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_t addr2 = {0};

    const char *listen_p = abcdk_option_get(args,"--listen",0,"0.0.0.0:12345");
    abcdk_sockaddr_from_string(&addr,listen_p,0);
    //addr.family = AF_UNIX;
    //strncpy(addr.addr_un.sun_path,sunpath,108);

    abcdk_comm_easy_t *easy_listen = abcdk_comm_easy_listen(ctx,server_ssl_ctx,&addr,test_easy_request_cb,NULL);

    const char *connect_p = abcdk_option_get(args,"--connect",0,"127.0.0.1:12345");
    abcdk_sockaddr_from_string(&addr2,connect_p,0);
    //addr2.family = AF_UNIX;
    //strncpy(addr2.addr_un.sun_path,sunpath,108);

    int nn = 4;
    abcdk_comm_easy_t *easy_client[40] = {NULL};
    for (int i = 0; i < nn; i++)
        easy_client[i] = abcdk_comm_easy_connect(ctx,client_ssl_ctx[i], &addr2, test_easy_request2_cb, NULL);

    uint64_t d = 0,s = 0;
    s = abcdk_clock(d,&d);

    

    #pragma omp parallel for num_threads(nn)
    for(int i = 0;i<100000;i++)
    {
#ifdef _OPENMP
        omp_get_thread_num();
#endif
        
        uint64_t d = 0,s = 0;
        s = abcdk_clock(d,&d);

        int len = 10000;
        char *req= (char*)abcdk_heap_alloc(len);
        abcdk_comm_message_t *rsp= NULL;

        sprintf(req,"%lu",abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6));

        abcdk_comm_easy_request(easy_client[i%nn],req,len,&rsp);
        

        if (rsp)
        {

            // printf("%d=%s\n",i,(char*)abcdk_comm_message_data(rsp));

            abcdk_comm_message_unref(&rsp);
        }
        else
        {
            printf("Pipe(%d) %s timeout\n",i%4,req);
        }

        abcdk_heap_free(req);

        s = abcdk_clock(d,&d);

      //  printf("[%d]:s = %lu,d = %lu\n",i,s,d);
    }

    s = abcdk_clock(d,&d);

    printf("s = %lu,d = %lu\n",s,d);

 //   abcdk_comm_easy_set_timeout(easy_listen,1);

  //  abcdk_comm_easy_unref(&easy_listen);
    
    for(int i = 0;i<nn;i++)
        abcdk_comm_easy_unref(&easy_client[i]);

    while (getchar() != 'Q')
        ;
    abcdk_comm_stop(&ctx);


}

int test_blkid(abcdk_tree_t *args)
{
#ifdef HAVE_BLKID
    int i, nparts;
	char *devname;
	blkid_probe pr;
	blkid_partlist ls;
	blkid_parttable root_tab;

	
	devname = (char*)abcdk_option_get(args,"--dev",0,"");

	pr = blkid_new_probe_from_filename(devname);
	if (!pr)
		return 1;

	/* Binary interface */
	ls = blkid_probe_get_partitions(pr);
	if (!ls)
		return 1;

	/*
	 * Print info about the primary (root) partition table
	 */
	root_tab = blkid_partlist_get_table(ls);
	if (!root_tab)
		return 1;

	printf("size: %jd, sector size: %u, PT: %s, offset: %jd, id=%s\n---\n",
		blkid_probe_get_size(pr),
		blkid_probe_get_sectorsize(pr),
		blkid_parttable_get_type(root_tab),
		blkid_parttable_get_offset(root_tab),
		blkid_parttable_get_id(root_tab));

	/*
	 * List partitions
	 */
	nparts = blkid_partlist_numof_partitions(ls);
	if (!nparts)
		goto done;

	for (i = 0; i < nparts; i++) {
		const char *p;
		blkid_partition par = blkid_partlist_get_partition(ls, i);
		blkid_parttable tab = blkid_partition_get_table(par);

		printf("#%d: %10llu %10llu  0x%x",
			blkid_partition_get_partno(par),
			(unsigned long long) blkid_partition_get_start(par),
			(unsigned long long) blkid_partition_get_size(par),
			blkid_partition_get_type(par));

		if (root_tab != tab)
			/* subpartition (BSD, Minix, ...) */
			printf(" (%s)", blkid_parttable_get_type(tab));

		p = blkid_partition_get_name(par);
		if (p)
			printf(" name='%s'", p);
		p = blkid_partition_get_uuid(par);
		if (p)
			printf(" uuid='%s'", p);
		p = blkid_partition_get_type_string(par);
		if (p)
			printf(" type='%s'", p);

		putc('\n', stdout);
	}

done:
	blkid_free_probe(pr);

#endif

	return EXIT_SUCCESS;
}

void test_bloom(abcdk_tree_t *args)
{

    size_t s = 1234567;
    uint8_t* buf = (uint8_t*)abcdk_heap_alloc(s);
#if 0
    for(size_t i = 0;i<s*8;i++)
    {
        assert(abcdk_bloom_mark(buf,s,i)==0);
    }

    for(size_t i = 0;i<s;i++)
    {
        assert(buf[i]==255);
    }
    

    for(size_t i = 0;i<s*8;i++)
    {
        assert(abcdk_bloom_filter(buf,s,i)==1);
        assert(abcdk_bloom_unset(buf,s,i)==0);
    }

    for(size_t i = 0;i<s;i++)
    {
        assert(buf[i]==0);
    }
#elif 0

    for (int i = 0; i < s * 8; i++)
        abcdk_bloom_write(buf, s, i, i % 2);

    for (int i = 0; i < s * 8; i++)
        assert(abcdk_bloom_read(buf, s, i) == i % 2);
#elif 0

    char dict[]={"ABCDEFGHJKLMNPQRSTUVWXYZ23456789"};

    int l = abcdk_align((12+2+8+2)*8,5);
    // for (int i = 0; i < l; i++)
    //     abcdk_bloom_write(buf, s, i, i % 2);

    abcdk_mac_fetch("eno1",ABCDK_PTR2I8PTR(buf,0));
    ABCDK_PTR2U16(buf,12) = 3<<8;
    ABCDK_PTR2U64(buf,14) = abcdk_time_clock2kind_with(CLOCK_REALTIME,0);
    ABCDK_PTR2U16(buf,22) = 65535;

     for (int i = 0; i < l;)
    {
        int v = 0, a = 0;
        for (int j = 0; j < 5; j++)
        {
            a = abcdk_bloom_read(buf, s, i++);
            v |= (a << (5 - j - 1));
        }
        printf("%c", dict[v]);
    }


    printf("\n");
#else
    
    char dict[]={"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="};

    int l = 12+2+8+2;
    int l2 = abcdk_align(l*8,6);
    int l3 = l2/8;

    abcdk_mac_fetch("eno1",ABCDK_PTR2I8PTR(buf,0));
    ABCDK_PTR2U16(buf,12) = 3<<8;
    ABCDK_PTR2U64(buf,14) = abcdk_time_clock2kind_with(CLOCK_REALTIME,0);
    ABCDK_PTR2U16(buf,22) = 65535;

    

    for (int i = 0; i < l2;)
    {
        int v = 0, a = 0;
        for (int j = 0; j < 6; j++)
        {
            a = abcdk_bloom_read(buf, s, i++);
            v |= (a << (6 - j - 1));
        }
        printf("%c", dict[v]);
    }

    for (int i = 0; i < 4-(l2 / 6 % 4); i++)
        printf("=");

    printf("\n");

    char buf2[200]={0};
    abcdk_base64_encode(buf,l3,buf2,200);
    //abcdk_base64_encode("qwer\n",5,buf2,200);

    printf("%s\n",buf2);

#endif

    abcdk_heap_free(buf);
}

void test_basecode(abcdk_tree_t *args)
{
    abcdk_basecode_t bctx = {0};

#if 1

    abcdk_basecode_init(&bctx, 32);

    char buf[100]={"abcdefghijklmnopqrstuvwxyz\n"};
    //char buf[100] = {"abc\n"};
    //char buf[100]={"a\n"};
    char buf2[100] = {0};

    //bctx.bit_align=1;
    int n = abcdk_basecode_encode(&bctx, buf, strlen(buf), buf2, 100);

    printf("n=%d,%s\n", n, buf2);

    char buf3[100] = {0};

    int m = abcdk_basecode_decode(&bctx, buf2, n, buf3, 100);

    printf("m = %d,%s\n", m, buf3);

#else

    abcdk_basecode_init(&bctx, 32);

    int l = 12 + 2 + 8 + 2;
    int l2 = abcdk_align(l,abcdk_math_lcm(8, 5)/8);

    char *buf = (char*)abcdk_heap_alloc(l2);
    char buf2[100] = {0};

    abcdk_mac_fetch("eno1",ABCDK_PTR2I8PTR(buf,0));
    ABCDK_PTR2U16(buf,12) = 3<<8;
    ABCDK_PTR2U64(buf,14) = abcdk_time_clock2kind_with(CLOCK_REALTIME,0);
    ABCDK_PTR2U16(buf,22) = 65535;

    abcdk_endian_swap(buf,l2);
    abcdk_cyclic_shift(buf,l2,3,2);
    abcdk_endian_swap(buf,l2);
    abcdk_cyclic_shift(buf,l2,3,2);
    abcdk_endian_swap(buf,l2);
    abcdk_cyclic_shift(buf,l2,3,2);

    int n = abcdk_basecode_encode(&bctx, buf, l2, buf2, 100);
    printf("n=%d,%s\n", n, buf2);

    char *buf3 = (char*)abcdk_heap_alloc(100);

    int m = abcdk_basecode_decode(&bctx, buf2, n, buf3, 100);

    assert(memcmp(buf,buf3,m)==0);

    abcdk_heap_free(buf3);

    abcdk_heap_free(buf);

#endif 

}

int test_clone_func(void *args)
{
    printf("child pid=%d\n",getpid());

#if 1

    const char root_path[] = {"/var/lib/docker/overlay2/b4f13ee76e071ee9278153556d34b4c9bd9061efd7627b4208156b68097390aa/merged"};

    chdir(root_path);
    chroot(root_path);

    int chk;
    
    // chk = mount("tmpfs", "/dev", "tmpfs", MS_NOSUID, "mode=0755");
    // chk = mkdir("/dev/pts", 0755);
    // assert(chk == 0);
    // chk = mkdir("/dev/socket", 0755);
    // assert(chk == 0);
    // chk = mount("devpts", "/dev/pts", "devpts", 0, NULL);
    // assert(chk == 0);
    // chk = mount("proc", "/proc", "proc", 0, NULL);
    // assert(chk == 0);
    // chk = mount("sysfs", "/sys", "sysfs", 0, NULL);
    // assert(chk == 0);

    char *argv[2] = {args, NULL};
    chk = execvp(argv[0], argv);
    printf("%d\n", errno);
    assert(chk == 0);
#else 

    for(int i = 0;i<1000;i++)
    {
        printf("i = %d\n",i);

        sleep(1);
    }

#endif

    return 0;
}

pid_t clone_wrapper(int (*func)(void *args), int flag, const char *cmd)
{
    printf("father pid=%d\n", getpid());

    int stack_size = 1024 * 1024;
    void *stack = malloc(stack_size);

    return clone(func, ABCDK_PTR2VPTR(stack, stack_size), flag, (void*)cmd);
}

void test_setns(abcdk_tree_t *args)
{
    int pid = abcdk_option_get_int(args, "--pid", 0, -1);
    const char *cmd = abcdk_option_get(args, "--cmd", 0, "/bin/bash");

    char buf[100] = {0};

    sprintf(buf, "/proc/%d/ns/pid", pid);

    int fd = abcdk_open(buf, 0, 0, 0);
    assert(fd >= 0);

    int chk = setns(fd, 0);
    assert(chk == 0);

    int fd2 = clone_wrapper(test_clone_func, SIGCHLD, cmd);
    assert(fd2 != 0);

    waitpid(fd2, NULL, 0);
}

void test_notify(abcdk_tree_t *args)
{
    const char* dir = abcdk_option_get(args,"--dir",0,"./");

    int fd = abcdk_notify_init(1);

    abcdk_notify_event_t t = {0};

    t.buf = abcdk_buffer_alloc2(4096);

    //int wd = abcdk_notify_add(fd,dir,IN_ALL_EVENTS);
    int wd = abcdk_notify_add(fd,dir,IN_CREATE|IN_DELETE|IN_MOVE_SELF|IN_MOVE);

    for(;;)
    {
        

        if(abcdk_notify_watch(fd,&t,-1)<0)
            break;

        if(t.event.mask & IN_ACCESS )
            printf("Access:");
        if(t.event.mask & IN_MODIFY )
            printf("Modify:");
        if(t.event.mask & IN_ATTRIB )
            printf("Metadata changed:");
        if(t.event.mask & IN_CLOSE )
            printf("Close:");
        if(t.event.mask & IN_OPEN )
            printf("Open:");
        if(t.event.mask & IN_MOVED_FROM )
            printf("Moved from(%u):",t.event.cookie);
        if(t.event.mask & IN_MOVED_TO )
            printf("Moved to(%u):",t.event.cookie);
        if(t.event.mask & IN_CREATE )
            printf("Created:");
        if(t.event.mask & IN_DELETE )
            printf("Deleted:");
        if(t.event.mask & IN_MOVE_SELF )
            printf("Deleted self:");
        if(t.event.mask & IN_UNMOUNT )
            printf("Umount:");
        if(t.event.mask & IN_IGNORED )
            printf("Ignored:");

        printf("%s\n",t.name);
    }

    abcdk_buffer_free(&t.buf);

    abcdk_closep(&fd);
}

void test_scsi(abcdk_tree_t *args)
{
    const char*dev = abcdk_option_get(args,"--dev",0,"./a");

    int fd = abcdk_open(dev,1,0,0);
    assert(fd>=0);

    uint8_t type = 255;
    char serial[255]={0};
    abcdk_scsi_io_stat_t stat = {0};
    abcdk_scsi_inquiry_serial(fd,&type,serial,0,&stat);

    uint8_t key = abcdk_scsi_sense_key(stat.sense);
    uint8_t asc = abcdk_scsi_sense_code(stat.sense);
    uint8_t ascq = abcdk_scsi_sense_qualifier(stat.sense);

    printf("key(%hhu),asc(%hhu),ascq(%hhu)\n",key,asc,ascq);

    printf("type(%hhu),serial(%s)\n",type,serial);

    abcdk_closep(&fd);
}


void test_fcgi(abcdk_tree_t *args)
{
#ifdef _FCGIAPP_H
    int chk;
    chk = FCGX_Init();

    int sock = FCGX_OpenSocket("127.0.0.1:9000",0);

    FCGX_Request request = {0};
    chk = FCGX_InitRequest(&request,sock,0);
    while (1)
    {
        chk = FCGX_Accept_r(&request);

        FCGX_FPrintF(request.out, "Content-Type: text/plain; charset=utf-8\r\n\r\n" );

        for(int i = 0;i<1000;i++)
        {
            if(!request.envp[i])
                break;

            FCGX_FPrintF(request.out,"[%d]=%s\r\n",i,request.envp[i]);

        }


        // char *a = FCGX_GetParam("REQUEST_METHOD",request.envp);
        // char *b = FCGX_GetParam("CONTENT_LENGTH",request.envp);

        FCGX_PutS("aaaaaaaaaaaaaaaaaaaaaa",request.out);



        FCGX_Finish_r(&request);
    }

#endif //_FCGIAPP_H
}

void test_geom(abcdk_tree_t *args)
{
#if 0
    abcdk_point_t *points;

    int numbers = 4;
    points = (abcdk_point_t*)abcdk_heap_alloc(numbers*sizeof(abcdk_point_t));

    points[0].x = 100;
    points[0].y = 100;
    points[1].x = 200;
    points[1].y = 100;
    points[2].x = 200;
    points[2].y = 200;
    points[3].x = 100;
    points[3].y = 200;

    abcdk_point_t b;
    b.x = abcdk_option_get_int(args,"--x",0,150);
    b.y = abcdk_option_get_int(args,"--y",0,150);

    int c = abcdk_point_in_polygon_2d(&b,points,numbers);
    printf("c=%d\n",c);

    abcdk_heap_free(points);
#elif 1

    double pi = 3.141592;
    abcdk_point_t a = {0},b = {0};
    abcdk_point_t a2 = {0},b2 = {0};

    a.x = 315;
    a.y = 151;
    b.x = 336;
    b.y = 140;

    double d = abcdk_line_length_3d(&a,&b);
    double r = abcdk_line_radian_2d(&a,&b,'x');
    double R = r*180.0/pi;
    printf("d = %lf,r = %lf(%lf)\n",d,r,R);

    abcdk_point_t ma = {0},mb = {0};

    abcdk_point_shift_2d(&a,r-90*pi/180,50,&ma);
    abcdk_point_shift_2d(&b,r-90*pi/180,50,&mb);

    double md = abcdk_line_length_3d(&ma,&mb);
    double mr = abcdk_line_radian_2d(&ma,&mb,'x');
    double mR = mr*180.0/pi;
    printf("d = %lf,r = %lf(%lf)\n",md,mr,mR);

    abcdk_point_t ma2 = {0},mb2 = {0};

    abcdk_point_shift_2d(&ma,r,-100,&ma2);
    abcdk_point_shift_2d(&mb,r,100,&mb2);

    double md2 = abcdk_line_length_3d(&ma2,&mb2);
    double mr2 = abcdk_line_radian_2d(&ma2,&mb2,'x');
    double mR2 = mr2*180.0/pi;
    printf("d = %lf,r = %lf(%lf)\n",md2,mr2,mR2);

  //  326 178 
  //  342 170
    abcdk_point_t p1,p2,p3,p4,p5;

    p1.x =0;
    p1.x =0;
    p2.x = 100;
    p2.y = 0;
    p3 = p1;
    p4 = p2;
    
    p3.x = 101;
    p3.y = 0;
    p4.x = 100;
    p4.y = 101;
    int  chk = abcdk_line_cross_2d(&p1,&p2,&p3,&p4,&p5);

    // ma2.x = (int)ma2.x;
    // ma2.y = (int)ma2.y;
    // mb2.x = (int)mb2.x;
    // mb2.y = (int)mb2.y;
    // int  chk = abcdk_line_cross_2d(&a,&b,&ma2,&mb2,&p5);

    printf("chk=%d\n",chk);
    printf("x=%lf,y=%lf\n",p5.x,p5.y);

#endif 
}

void test_fb(abcdk_tree_t *args)
{
    int fd = 0;
    struct fb_var_screeninfo vinfo = {0};
    struct fb_fix_screeninfo finfo = {0};

    fd = abcdk_open("/dev/fb0",1,0,0);

    ioctl(fd, FBIOGET_FSCREENINFO, &finfo);
    ioctl(fd, FBIOGET_VSCREENINFO, &vinfo);

    printf("%dx%d, %dbpp\n", vinfo.xres, vinfo.yres, vinfo.bits_per_pixel );
    printf("%dx%d\n", vinfo.xres_virtual, vinfo.yres_virtual);

    long screen_size = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel/8;

    void *p = mmap(NULL,screen_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);

    vinfo.xoffset = 0;
    vinfo.yoffset = 0;

    ioctl(fd,FBIOPAN_DISPLAY,&vinfo);

    sleep(1);

    for(int i = 0;i<vinfo.yres;i++)
    for(int j = 0;j<vinfo.xres;j++)
        ABCDK_PTR2U8(p,i*vinfo.xres*3+j*3) = 0xff;

    vinfo.xoffset = 0;
    vinfo.yoffset = vinfo.yres;

    ioctl(fd,FBIOPAN_DISPLAY,&vinfo);

    sleep(1);


    abcdk_closep(&fd);
}

void test_udev(abcdk_tree_t *args)
{
#ifdef _LIBUDEV_H_
    struct udev *udev;
    struct udev_enumerate *enumerate;
    struct udev_list_entry *devices, *dev_list_entry;
    struct udev_device *dev;

    /* Create the udev object */
    udev = udev_new();
    if (!udev)
    {
        printf("Can't create udev\n");
        exit(1);
    }

    /* Create a list of the devices in the 'hidraw' subsystem. */
    enumerate = udev_enumerate_new(udev);
    //udev_enumerate_add_match_subsystem(enumerate, "class");
    //udev_enumerate_scan_devices(enumerate);
    udev_enumerate_scan_subsystems(enumerate);
    devices = udev_enumerate_get_list_entry(enumerate);
    /* For each item enumerated, print out its information.
       udev_list_entry_foreach is a macro which expands to
       a loop. The loop will be executed for each member in
       devices, setting dev_list_entry to a list entry
       which contains the device's path in /sys. */
    udev_list_entry_foreach(dev_list_entry, devices)
    {
        const char *path;

        /* Get the filename of the /sys entry for the device
           and create a udev_device object (dev) representing it */
        path = udev_list_entry_get_name(dev_list_entry);
        dev = udev_device_new_from_syspath(udev, path);

        if(strstr(path,"bus")==NULL)
            continue;
        printf("\n%s\n",path);

        /* usb_device_get_devnode() returns the path to the device node
           itself in /dev. */
        printf("Device Node Path: %s\n", udev_device_get_devnode(dev));

        /* The device pointed to by dev contains information about
           the hidraw device. In order to get information about the
           USB device, get the parent device with the
           subsystem/devtype pair of "usb"/"usb_device". This will
           be several levels up the tree, but the function will find
           it.*/
        dev = udev_device_get_parent_with_subsystem_devtype(
            dev,
            "usb",
            "usb_device");
        if (!dev)
        {
            printf("Unable to find parent usb device.");
            continue;
        }

        /* From here, we can call get_sysattr_value() for each file
           in the device's /sys entry. The strings passed into these
           functions (idProduct, idVendor, serial, etc.) correspond
           directly to the files in the directory which represents
           the USB device. Note that USB strings are Unicode, UCS2
           encoded, but the strings returned from
           udev_device_get_sysattr_value() are UTF-8 encoded. */
        printf(" VID/PID: %s %s\n",
               udev_device_get_sysattr_value(dev, "idVendor"),
               udev_device_get_sysattr_value(dev, "idProduct"));
        printf(" %s\n %s\n",
               udev_device_get_sysattr_value(dev, "manufacturer"),
               udev_device_get_sysattr_value(dev, "product"));
        printf(" serial: %s\n",
               udev_device_get_sysattr_value(dev, "serial"));
        udev_device_unref(dev);
    }
    /* Free the enumerator object */
    udev_enumerate_unref(enumerate);

    udev_unref(udev);

#endif //_LIBUDEV_H_
}

void test_dmtx(abcdk_tree_t *args)
{
#if defined(HAVE_LIBDMTX) && defined(HAVE_FREEIMAGE)
    abcdk_fi_init(0);

    DmtxImage *img = NULL;
    DmtxDecode *dec = NULL;
    DmtxRegion *reg = NULL;
    DmtxMessage *msg = NULL;

    const char *file = abcdk_option_get(args, "--img", 0, "");

    FREE_IMAGE_FORMAT file_fmt = FreeImage_GetFileType(file,0);
    FIBITMAP *fi = abcdk_fi_load2(file_fmt, 0, file);

    uint8_t*ptr = FreeImage_GetBits(fi);
    int width = FreeImage_GetWidth(fi);
    int height = FreeImage_GetHeight(fi);
    int channels = FreeImage_GetBPP(fi) / 8;
    FREE_IMAGE_COLOR_TYPE type = FreeImage_GetColorType(fi);

    if(type == FIC_RGB)
    img = dmtxImageCreate(ptr, width, height, DmtxPack24bppRGB);
    else if(type == FIC_RGBALPHA)
        img = dmtxImageCreate(ptr, width, height, DmtxPack32bppRGBX);
    dec = dmtxDecodeCreate(img, 1);
    reg = dmtxRegionFindNext(dec, NULL);
    msg = dmtxDecodeMatrixRegion(dec, reg, DmtxModuleOnRGB);
    //msg = dmtxDecodeMosaicRegion(dec, reg,DmtxUndefined);

    printf("{%s}\n",msg->output);
    

    dmtxMessageDestroy(&msg);
    dmtxRegionDestroy(&reg);
    dmtxDecodeDestroy(&dec);
    dmtxImageDestroy(&img);
    FreeImage_Unload(fi);
    abcdk_fi_uninit();

#endif
}

void test_zbar(abcdk_tree_t *args)
{
#ifdef HAVE_ZBAR

#ifdef HAVE_FREEIMAGE
    abcdk_fi_init(0);

    const char *file = abcdk_option_get(args, "--img", 0, "");

    FREE_IMAGE_FORMAT file_fmt = FreeImage_GetFileType(file, 0);
    FIBITMAP *fi = abcdk_fi_load2(file_fmt, 0, file);

    uint8_t *ptr = FreeImage_GetBits(fi);
    int width = FreeImage_GetWidth(fi);
    int height = FreeImage_GetHeight(fi);
    int channels = FreeImage_GetBPP(fi) / 8;
    int pitch = FreeImage_GetPitch(fi);
    FREE_IMAGE_COLOR_TYPE type = FreeImage_GetColorType(fi);
    

    zbar_image_t *image = zbar_image_create();
    //zbar_image_set_format(image,ABCDK_FOURCC_MKTAG('G','R','E','Y'));
    zbar_image_set_format(image, ABCDK_FOURCC_MKTAG('Y', '8', '0', '0'));
    zbar_image_set_size(image, width, height);

#if 1 
    uint8_t *data = (uint8_t*)abcdk_heap_alloc(width * height);

    abcdk_ndarray_t src = {0} ,dst = {0};
    src.fmt = ABCDK_NDARRAY_NHWC;
    src.blocks =1;
    src.width = width;
    src.height = height;
    src.depth = channels;
    src.width_bytes = pitch;
    src.cell_bytes = 1;

    dst.fmt = ABCDK_NDARRAY_NHWC;
    dst.blocks = 1;
    dst.width = width;
    dst.height = height;
    dst.depth = 1;
    dst.width_bytes = width;
    dst.cell_bytes = 1;

    uint8_t *tmp = ptr;
    uint8_t *tmp2 = data;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            size_t r_pos = abcdk_ndarray_offset(&src,0,x,y,0,0);
            size_t g_pos = abcdk_ndarray_offset(&src,0,x,y,1,0);
            size_t b_pos = abcdk_ndarray_offset(&src,0,x,y,2,0);
            uint8_t r = tmp[r_pos], g = tmp[g_pos], b = tmp[b_pos];

             uint8_t gray = (r * 38 + g * 75 + b * 15) >> 7;
            //uint8_t gray = r * 0.299 + g * 0.587 + b * 0.114;
 
            size_t gray_pos = abcdk_ndarray_offset(&dst,0,x,y,0,ABCDK_NDARRAY_FLIP_V);//FreeImage 解码的图像头下脚上(没有自动翻转)，这里转灰度时要翻转一下。
            tmp2[gray_pos] = gray;
        }
        

    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            size_t gray_pos = abcdk_ndarray_offset(&dst, 0, x, y, 0, 0);
            printf("%c", tmp2[gray_pos] > 128 ? '.' : ' ');
        }
        printf("\n");
    }

    zbar_image_set_data(image, data, width * height, NULL);

#else 

    zbar_image_set_data(image, ptr, width * height, NULL);
#endif 

    zbar_image_scanner_t *scaner = zbar_image_scanner_create();
    zbar_image_scanner_set_config(scaner, ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
    int chk = zbar_scan_image(scaner, image);
   // const zbar_symbol_set_t *sym = zbar_image_scanner_get_results(scaner);

   // zbar_processor_t *processor = zbar_processor_create(0);
   // zbar_processor_init(processor, NULL, 1);

   // zbar_process_image(processor, image);

    const zbar_symbol_t *sym = zbar_image_first_symbol(image);
    for (; sym; sym = zbar_symbol_next(sym))
    {
        zbar_symbol_type_t typ = zbar_symbol_get_type(sym);
        if (typ == ZBAR_PARTIAL)
            continue;
        else
            printf("%s%s:%s\n", zbar_get_symbol_name(typ), zbar_get_addon_name(typ), zbar_symbol_get_data(sym));
    }

   // zbar_image_scanner_destroy(scaner);
    zbar_image_destroy(image);
    FreeImage_Unload(fi);
    abcdk_fi_uninit();
#elif defined(HAVE_MAGICKWAND)

    static int notfound = 0, exit_code = 0;
    static int num_images = 0, num_symbols = 0;
    static int xmllvl = 0;

    char *xmlbuf = NULL;
    unsigned xmlbuflen = 0;

    static zbar_processor_t *processor = NULL;

    const char *filename = abcdk_option_get(args, "--img", 0, "");

    MagickWandGenesis();

    processor = zbar_processor_create(0);

    zbar_processor_init(processor, NULL, 1);

    int found = 0;
    MagickWand *images = NewMagickWand();
    if (!MagickReadImage(images, filename))
        return ;

    unsigned seq, n = MagickGetNumberImages(images);
    for (seq = 0; seq < n; seq++)
    {
        if (exit_code == 3)
            return;

        if (!MagickSetIteratorIndex(images, seq))
            return;

        zbar_image_t *zimage = zbar_image_create();
        assert(zimage);
        zbar_image_set_format(zimage, *(unsigned long *)"Y800");

        int width = MagickGetImageWidth(images);
        int height = MagickGetImageHeight(images);
        zbar_image_set_size(zimage, width, height);

        // extract grayscale image pixels
        // FIXME color!! ...preserve most color w/422P
        // (but only if it's a color image)
        size_t bloblen = width * height;
        unsigned char *blob = malloc(bloblen);
        zbar_image_set_data(zimage, blob, bloblen, zbar_image_free_data);

        if (!MagickExportImagePixels(images, 0, 0, width, height,
                                     "I", CharPixel, blob))
            return;

        uint8_t *tmp2 = blob;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                printf("%c", tmp2[x] > 128 ? ' ' : '.');
            }
            printf("\n");
            tmp2 += width;
        }

        if (xmllvl == 1)
        {
            xmllvl++;
            printf("<source href='%s'>\n", filename);
        }

        zbar_process_image(processor, zimage);

        // output result data
        const zbar_symbol_t *sym = zbar_image_first_symbol(zimage);
        for (; sym; sym = zbar_symbol_next(sym))
        {
            zbar_symbol_type_t typ = zbar_symbol_get_type(sym);
            if (typ == ZBAR_PARTIAL)
                continue;
            else if (!xmllvl)
                printf("%s%s:%s\n",
                       zbar_get_symbol_name(typ),
                       zbar_get_addon_name(typ),
                       zbar_symbol_get_data(sym));
            else if (xmllvl < 0)
                printf("%s\n", zbar_symbol_get_data(sym));
            else
            {
                if (xmllvl < 3)
                {
                    xmllvl++;
                    printf("<index num='%u'>\n", seq);
                }
                zbar_symbol_xml(sym, &xmlbuf, &xmlbuflen);
                printf("%s\n", xmlbuf);
            }
            found++;
            num_symbols++;
        }
        if (xmllvl > 2)
        {
            xmllvl--;
            printf("</index>\n");
        }
        fflush(stdout);

        zbar_image_destroy(zimage);

        num_images++;
        if (zbar_processor_is_visible(processor))
        {
            int rc = zbar_processor_user_wait(processor, -1);
            if (rc < 0 || rc == 'q' || rc == 'Q')
                exit_code = 3;
        }
    }

    if (xmllvl > 1)
    {
        xmllvl--;
        printf("</source>\n");
    }

    if (!found)
        notfound++;

    DestroyMagickWand(images);
#endif
#endif
}

void test_ndarray(abcdk_tree_t *args)
{
    abcdk_ndarray_t src = {0}, dst = {0};
    src.fmt = ABCDK_NDARRAY_NCHW;
    src.blocks = 2;
    src.width = 10;
    src.height = 10;
    src.depth = 3;
    src.cell_bytes = 2;
    abcdk_ndarray_set_width_bytes(&src, 4);
    src.data = abcdk_heap_alloc(abcdk_ndarray_size(&src));

    dst.fmt = ABCDK_NDARRAY_NHWC;
    dst.blocks = 2;
    dst.width = 10;
    dst.height = 10;
    dst.depth = 3;
    dst.cell_bytes = 2;
    abcdk_ndarray_set_width_bytes(&dst, 1);
    dst.data = abcdk_heap_alloc(abcdk_ndarray_size(&dst));

    for (int n = 0; n < src.blocks; n++)
    {
        for (int y = 0; y < src.height; y++)
        {
            for (int x = 0; x < src.width; x++)
            {
                for (int z = 0; z < src.depth; z++)
                {
                    size_t pos_src = abcdk_ndarray_offset(&src, n, x, y, z, 0);
                    ABCDK_PTR2U16(src.data, pos_src) = z+1;
                }
            }
        }
    }

    for (int n = 0; n < src.blocks; n++)
    {
        for (int y = 0; y < src.height; y++)
        {
            for (int x = 0; x < src.width; x++)
            {
                for (int z = 0; z < src.depth; z++)
                {
                    size_t pos_src = abcdk_ndarray_offset(&src, n, x, y, z, 0);
                    size_t pos_dst = abcdk_ndarray_offset(&dst, n, x, y, z, 0);
                    ABCDK_PTR2U16(dst.data, pos_dst) = ABCDK_PTR2U16(src.data, pos_src);
                }
            }
        }
    }


    for (int n = 0; n < src.blocks; n++)
    {
        for (int y = 0; y < src.height; y++)
        {
            for (int x = 0; x < src.width; x++)
            {
                for (int z = 0; z < src.depth; z++)
                {
                    //size_t pos_src = abcdk_ndarray_offset(&src, n, x, y, z, 0);
                    //printf("%hu",ABCDK_PTR2U16(src.data, pos_src));
                    size_t pos_dst = abcdk_ndarray_offset(&dst, n, x, y, z, 0);
                    printf("%hu",ABCDK_PTR2U16(dst.data, pos_dst));
                }
                printf("\n");
            }
        }
    }

}

void test_unix_sock(abcdk_tree_t *args)
{
    const char *sunpath = "/tmp/test_unix_sock.sock";

    int isserver = abcdk_option_exist(args,"--isserver");

    if (isserver)
    {
        unlink(sunpath);

        abcdk_sockaddr_t addr = {0};
        addr.family = AF_UNIX;
        strncpy(addr.addr_un.sun_path, sunpath, 108);

        int sock_listen = abcdk_socket(AF_UNIX, 0);
        
        abcdk_bind(sock_listen, &addr);
        listen(sock_listen, 5);
        //fchmod(sock_listen,0777);
       // chmod(addr.addr_un.sun_path,0777);

        while(1)
        {
            abcdk_sockaddr_t addr2 = {0};
            int sock = abcdk_accept(sock_listen,&addr2);
            if(sock<0)
                break;

            while (1)
            {
                char buf[100];
                int r = recv(sock, buf, 99, 0);
                if (r <= 0)
                    break;

                printf("{%s}\n",buf);

                send(sock,"2000",4,0);
            }

            abcdk_closep(&sock);
        }

        abcdk_closep(&sock_listen);
    }
    else 
    {
        abcdk_sockaddr_t addr = {0};
        addr.family = AF_UNIX;
        strncpy(addr.addr_un.sun_path, sunpath, 108);

        int sock = abcdk_socket(AF_UNIX, 0);

        int chk = abcdk_connect(sock,&addr,1000);
        if(chk==0)
        {
            send(sock,"1000",4,0);

            char buf[100];
            recv(sock,buf,100,0);

            printf("{%s}\n",buf);
        }

        abcdk_closep(&sock);
    }
}

void test_mtab(abcdk_tree_t *args)
{
    abcdk_tree_t *list = abcdk_tree_alloc3(1);

    abcdk_mtab_list(list);

    abcdk_tree_t *p;

    p = abcdk_tree_child(list,1);
    while(p)
    {
        abcdk_mtab_info_t *dev_p  = (abcdk_mtab_info_t*)p->alloc->pptrs[0];

        struct statfs s = {0};
        statfs(dev_p->mpoint,&s);

        printf("%s %s %lu %lu\n",dev_p->fs,dev_p->mpoint,s.f_bsize*s.f_blocks,s.f_bsize*s.f_bavail);

        p = abcdk_tree_sibling(p,0);
    }

    abcdk_tree_free(&list);
}

void test_block(abcdk_tree_t *args)
{
    const char *name = abcdk_option_get(args,"--name",0,"");

    char path[PATH_MAX] = {0};
    int chk = abcdk_block_find_device(name,path);
    assert(chk == 0);

    abcdk_mmc_info_t mmc = {0};
    abcdk_scsi_info_t scsi = {0};

    chk = abcdk_mmc_get_info(path,&mmc);
    if(chk != 0)
    {
        chk = abcdk_scsi_get_info(path,&scsi);
        printf("scsi:%s,%s\n", scsi.serial ,scsi.devname);
    }
    else
    {
        printf("mmc:%s,%s\n", mmc.cid,mmc.devname);
    }

}


void test_sqlite(abcdk_tree_t *args)
{
#if defined(_SQLITE3_H_) || defined(SQLITE3_H)

    const char *name = abcdk_option_get(args,"--name",0,"/tmp/aaaa.sqlite");

    sqlite3* ctx = abcdk_sqlite_open(name);

    char *sql[] = {"CREATE TABLE files (file_number INTEGER NOT NULL,file_path_name TEXT(4096) NOT NULL,file_hash_code TEXT(256),file_hash_type INTEGER,file_status INTEGER NOT NULL);",
    "CREATE UNIQUE INDEX files_file_number_IDX ON files (file_number);",
    "CREATE INDEX files_file_status_IDX ON files (file_status);",
    "CREATE UNIQUE INDEX files_file_path_name_IDX ON files (file_path_name);",
    "CREATE INDEX files_file_number_status_IDX ON files (file_number,file_status);",
    "CREATE TABLE volumes (volume_number INTEGER NOT NULL,volume_name TEXT(4096) NOT NULL,volume_status INTEGER NOT NULL);",
    "CREATE UNIQUE INDEX volumes_volume_number_IDX ON volumes (volume_number);",
    "CREATE UNIQUE INDEX volumes_volume_name_IDX ON volumes (volume_name);",
    "CREATE INDEX volumes_volume_status_IDX ON volumes (volume_status);",
    "CREATE TABLE indexes (file_number INTEGER NOT NULL,volume_number INTEGER NOT NULL,block_offset INTEGER NOT NULL,file_offset INTEGER NOT NULL);",
    "CREATE INDEX indexes_file_number_vol_number_IDX ON indexes (file_number,volume_number);",
    NULL};

    abcdk_sqlite_tran_begin(ctx);

    for(int i = 0;sql[i];i++)
        abcdk_sqlite_exec_direct(ctx,sql[i]);

    abcdk_sqlite_tran_commit(ctx);

    abcdk_sqlite_tran_begin(ctx);

    char buf[1000] = {0};
    for(int i = 0;i<30000;i++)
    {
        memset(buf,0,1000);
        sprintf(buf,"insert into files(file_number,file_path_name,file_hash_code,file_hash_type,file_status) "
             " values('%d','aaaaaaaaaaaaaa000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000aaaa_%d','bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb_%d','%d','%d');",i,i,i,1,1);
        abcdk_sqlite_exec_direct(ctx,buf);

        for(int j=0;j<8;j++)
        {   memset(buf,0,1000);
            sprintf(buf,"insert into indexes(file_number,volume_number,block_offset,file_offset) "
            " values('%d','%d','%d','%d');",i,j,i,i);
        

     //   printf("%s\n",buf);
        abcdk_sqlite_exec_direct(ctx,buf);
        }
    }

    abcdk_sqlite_tran_commit(ctx);

    abcdk_sqlite_close(ctx);

#endif //
}

void test_reader(abcdk_tree_t *args)
{
    const char *src = abcdk_option_get(args,"--src",0,"/dev/zero");
    const char *dst = abcdk_option_get(args,"--dst",0,"/tmp/test_reader.data");
    long blksize = abcdk_option_get_long(args,"--blksize",0,256*1024);
    long bufsize = abcdk_option_get_long(args,"--bufsize",0,256*1024);

    abcdk_reader_t *r = abcdk_reader_create(blksize);

    int fd_src = abcdk_open(src,0,0,0);
    int fd_dst = abcdk_open(dst,1,0,1);

    abcdk_reader_start(r,fd_src);
    abcdk_reader_start(r,fd_src);

    char *buf = (char*)abcdk_heap_alloc(bufsize);
    for(;;)
    {
        int n = abcdk_reader_read(r,buf,bufsize);
        if(n<=0)
            break;
        
        int m = abcdk_write(fd_dst,buf,n);
        assert(m==n);
        //printf("[%d]%d\n",i,n);
    }
    
    abcdk_heap_free(buf);
    
    abcdk_reader_stop(r);
    abcdk_reader_stop(r);

    abcdk_closep(&fd_dst);
    abcdk_closep(&fd_src);

    abcdk_reader_destroy(&r);
}

#ifdef RD_KAFKA_VERSION

void test_kafka_consumer()
{
    char errstr[1001] = {0};

    int chk;

    rd_kafka_conf_t *conf = rd_kafka_conf_new();
    /* Quick termination */
    char tmp[16] = {0};
    snprintf(tmp, sizeof(tmp), "%i", SIGIO);
    rd_kafka_conf_set(conf, "internal.termination.signal", tmp, NULL, 0);
    rd_kafka_conf_set(conf, "message.timeout.ms", "3000", NULL, 0);
    rd_kafka_conf_set(conf, "socket.timeout.ms", "3000", NULL, 0);
    rd_kafka_conf_set(conf, "socket.keepalive.enable", "true", NULL, 0);
    rd_kafka_conf_set(conf, "group.id", "aaaa", NULL, 0);

    rd_kafka_t *k = rd_kafka_new(RD_KAFKA_CONSUMER,conf,errstr,1000);
    chk = rd_kafka_brokers_add(k,"192.167.200.102:9092");

    rd_kafka_topic_conf_t *topic_conf = rd_kafka_topic_conf_new();

    rd_kafka_topic_t *topic = rd_kafka_topic_new(k, "aaaa", topic_conf);
    int partition = 0;
    chk = rd_kafka_consume_start(topic,partition,RD_KAFKA_OFFSET_END);

    while (1)
    {
        rd_kafka_message_t *rkmessage;
        rd_kafka_resp_err_t err;

        /* Poll for errors, etc. */
        rd_kafka_poll(k, 0);

        rkmessage = rd_kafka_consume(topic, partition, 1000);
        if (!rkmessage)
            continue;

        if (rkmessage->len > 0)
        {
            char *key = NULL;
            char *val = abcdk_heap_clone(rkmessage->payload, rkmessage->len);
            if (rkmessage->key_len > 0)
                key = abcdk_heap_clone(rkmessage->key, rkmessage->key_len);
            abcdk_log_printf(LOG_DEBUG, "key(%s),val(%s)", key, val);
            abcdk_heap_free(val);
            abcdk_heap_free2((void **)&key);
        }

        rd_kafka_message_destroy(rkmessage);
    }

    /* Stop consuming */
    rd_kafka_consume_stop(topic, partition);
    
    while (rd_kafka_outq_len(k) > 0)
            rd_kafka_poll(k, 10);
    
    /* Destroy topic */
    rd_kafka_topic_destroy(topic);

    rd_kafka_destroy(k);

}


void test_kafka_producer()
{
    char errstr[1001] = {0};

    int chk;

    rd_kafka_conf_t *conf = rd_kafka_conf_new();
    /* Quick termination */
    char tmp[16] = {0};
    snprintf(tmp, sizeof(tmp), "%i", SIGIO);
    rd_kafka_conf_set(conf, "internal.termination.signal", tmp, NULL, 0);
    rd_kafka_conf_set(conf, "message.timeout.ms", "3000", NULL, 0);
    rd_kafka_conf_set(conf, "socket.timeout.ms", "3000", NULL, 0);
    rd_kafka_conf_set(conf, "socket.keepalive.enable", "true", NULL, 0);
    rd_kafka_conf_set(conf, "group.id", "aaaa", NULL, 0);

    rd_kafka_t *k = rd_kafka_new(RD_KAFKA_PRODUCER,conf,errstr,1000);
    chk = rd_kafka_brokers_add(k,"192.167.200.102:9092");

    rd_kafka_topic_conf_t *topic_conf = rd_kafka_topic_conf_new();

    rd_kafka_topic_t *topic = rd_kafka_topic_new(k, "aaaa", topic_conf);
    int partition = 0;
    

    for(int i = 0;i<1000000;i++)
    {
        char key[100] = {0};
        char val[100] = {0};

        sprintf(key,"key-%d",i);
        sprintf(val,"val-%d",i);

        chk = rd_kafka_produce(topic,partition,RD_KAFKA_MSG_F_COPY,val,strlen(val),key,strlen(key),NULL);

        if(chk !=0 )
        {
            rd_kafka_resp_err_t kaerrno = rd_kafka_errno2err(errno);
            abcdk_log_printf(LOG_DEBUG, "%% Failed to produce to topic %s partition %i: %s\n",
                    rd_kafka_topic_name(topic), partition, rd_kafka_err2str(kaerrno));

            if(kaerrno != RD_KAFKA_RESP_ERR__QUEUE_FULL)
                break;

            rd_kafka_poll(k, 1000);
            i--;
        }

        rd_kafka_poll(k, 0);
    }

     rd_kafka_poll(k, 0);

    
    while (rd_kafka_outq_len(k) > 0)
            rd_kafka_poll(k, 10);
    
    /* Destroy topic */
    rd_kafka_topic_destroy(topic);

    rd_kafka_destroy(k);

}


#endif //RD_KAFKA_VERSION

void test_kafka(abcdk_tree_t *args)
{
    int role = abcdk_option_get_int(args,"--role",0,1);
#ifdef RD_KAFKA_VERSION

    abcdk_log_printf(LOG_INFO,"%s",rd_kafka_version_str());

#if 0 
    if(role)
        test_kafka_consumer();
    else 
        test_kafka_producer();
#else 
    #pragma omp parallel for num_threads(2)
    for(int i = 0;i<2;i++)
    {
        if(i == 0)
            test_kafka_consumer();
        else if (i == 1)
            test_kafka_producer();
    }
#endif 

#endif //RD_KAFKA_VERSION
}


void test_record(abcdk_tree_t *args)
{
    const char *src = abcdk_option_get(args,"--src",0,"");
    const char *dst = abcdk_option_get(args,"--dst",0,"/tmp/");

     //-map 0:v  -dframes 5 -f rawvideo -pix_fmt rgb24  pipe:1 
    //char cmd[] = {"ffmpeg -y -rtsp_transport tcp -i rtsp://admin:a1234567@192.167.10.106:554 -map 0:v -vf select='eq(pict_type\\,I)' -vsync 2 -f rawvideo -pix_fmt rgb24  pipe:1 -c:v copy -f segment -segment_format mp4 -segment_time 5 -reset_timestamps 1 /tmp/bbbb/abcdk_%d.mp4"};

    //ffmpeg -y -pattern_type glob -i "images/*.jpg" -r 2 -vc h264 -f mp4 video.mp4
    char cmd[1024] = {0};
  //  sprintf(cmd,"ffmpeg -y -rtsp_transport tcp -i %s -map 0:v -vf select='eq(pict_type\\,I)' -vsync 2 -f rawvideo -pix_fmt rgb24  pipe:1 -c:v copy -f segment -segment_format mp4 -segment_time 5 -reset_timestamps 1 %s/%%d.mp4",
    sprintf(cmd,"ffmpeg -y -rtsp_transport tcp -stimeout 5000000 -i %s -c:v copy -c:a copy -f segment -segment_format mp4 -segment_time 60 -reset_timestamps 1 %s/%%09d.mp4",
        src,dst);

    printf("%s\n",cmd);

    pid_t p = abcdk_popen(cmd,NULL,NULL,NULL,NULL);
#if 0
    int out = -1;
    pid_t p = abcdk_popen(cmd,NULL,NULL,&out,NULL);

    size_t bsize = 1920*1080*3;
    void *buf = abcdk_heap_alloc(bsize);

    for(int i = 0;i<10000;i++)
    {
        ssize_t r = abcdk_read(out,buf,bsize);
        if(r != bsize)
            continue;

        char file[100] = {0};
        //sprintf(file,"/tmp/bbbb/bmp/%02X/%04d.bmp",i/256,i);
        sprintf(file,"/tmp/bbbb/bmp/%04d.bmp",i);

  //      abcdk_mkdir(file,0700);
  //      abcdk_bmp_save2(file,buf,1920*3,1920,1080,24);

    }
    abcdk_heap_free(buf);
    abcdk_closep(&out);

    kill(p,SIGTERM);
#endif
    int status;
    waitpid(p,&status,0);

    printf("%d\n",WEXITSTATUS(status));
}

void test_dup(abcdk_tree_t *args)
{
    int fd = abcdk_socket(AF_INET,0);

    abcdk_sockaddr_t a = {0};
    abcdk_sockaddr_from_string(&a,"www.baidu.com:443",1);
    abcdk_connect(fd,&a,5000);

    int fd2 = dup2(fd,1);
    abcdk_closep(&fd);


    struct pollfd arr = {0};
    arr.fd = fd2;
    arr.events = POLLERR |POLLHUP|POLLNVAL;
    poll(&arr,1,-1);

    abcdk_closep(&fd2);
}

void test_file_wholockme(abcdk_tree_t *args)
{
    const char *file = abcdk_option_get(args,"--file",0,"");

    int pids[1000] = {0};

    int c = abcdk_file_wholockme(file,pids,1000);

    for (int i = 0; i < c; i++)
    {
        printf("pid:%d\n",pids[i]);
    }
}

void test_file_segment(abcdk_tree_t *args)
{
    const char *file = abcdk_option_get(args,"--file",0,"");
    const char *fmt = abcdk_option_get(args,"--fmt",0,"%d");
    int max = abcdk_option_get_int(args,"--max",0,6);

    for(int i = 0;i<100;i++)
    {
        abcdk_save(file,&i,4,0);
        abcdk_file_segment(file,fmt,max);
    }
}


#ifdef HAVE_UNIXODBC
int test_odbcpool_connect(abcdk_odbc_t *odbc,void *opaque)
{
    abcdk_tree_t *args = (abcdk_tree_t *)opaque;

    const char *product = abcdk_option_get(args, "--product", 0, "");
    const char *driver = abcdk_option_get(args, "--driver", 0, "");
    const char *host = abcdk_option_get(args, "--host", 0, "localhost");
    uint16_t port = abcdk_option_get_int(args, "--port", 0,12345);
    const char *db = abcdk_option_get(args, "--db", 0, "");
    const char *user = abcdk_option_get(args, "--user", 0, "");
    const char *pwd = abcdk_option_get(args, "--pwd", 0, "");
    time_t timeout = abcdk_option_get_long(args, "--timeout", 0, 30);
    const char *tracefile = abcdk_option_get(args, "--tracefile", 0, NULL);

    SQLRETURN ret = abcdk_odbc_connect2(odbc,product,driver,host,port,db,user,pwd,timeout,tracefile);
    if(ret == SQL_SUCCESS)
        return 0;
    
    return -1;

}
#endif

void test_odbcpool(abcdk_tree_t *args)
{
#ifdef HAVE_UNIXODBC
    abcdk_odbcpool_t *h = abcdk_odbcpool_create(10,test_odbcpool_connect,args);

    #pragma omp parallel for num_threads(30)
    for (int i = 0; i < 100; i++)
    {
        printf("[%d]\n",i);
        abcdk_odbc_t *odbc = abcdk_odbcpool_pop(h,10*1000);
        usleep(rand() % 1000000);
        abcdk_odbcpool_push(h, &odbc);
    }

    abcdk_odbcpool_destroy(&h);
#endif
}

void test_log(abcdk_tree_t *args)
{
    abcdk_log_mask(1,2,3,4,-1);

    for(int j =0;j<ABCDK_LOG_MAX;j++)
    {
        for (int i = 0; i < 1000; i++)
            abcdk_log_printf(-j, "%d", j);
    }
}

int main(int argc, char **argv)
{
    abcdk_thread_t p;
    p.routine = sigwaitinfo_cb;
    abcdk_thread_create(&p,0);

    srand(time(NULL));

    abcdk_tree_t *args = abcdk_tree_alloc3(1);
    abcdk_getargs(args,argc,argv,"--");
    
    abcdk_option_fprintf(stderr,args,NULL);

    const char *func = abcdk_option_get(args,"--func",0,"");
    

   // abcdk_clock_reset();
#if 0
    int a = 0x112233;
    int b = 0;
    char a8[3] = {0};

    abcdk_endian_h_to_b24(a8,a);;
    b = abcdk_endian_b_to_h24(a8);
    assert(a == b);

    abcdk_endian_h_to_l24(a8,a);
    b = abcdk_endian_l_to_h24(a8);
    assert(a == b);

    uint64_t c = 1234567890987654321;
    uint64_t d = 0,e = 0;
    d = abcdk_endian_h_to_b64(c);
    e = abcdk_endian_b_to_h64(d);
    assert(c == e);

    uint64_t f = 0,g = 0;
    g = abcdk_clock(f,&f);
    for(int i = 0;i<100000;i++)
    {
        void *p = abcdk_heap_alloc(1024);
        abcdk_heap_free(p);
    }
    g = abcdk_clock(f,&f);
    printf("g = %lu,f = %lu\n",g,f);

    for (int i = 0; i < 10000; i++)
    {
        int v = rand() % 127;
        int k = ABCDK_CLAMP(v,33,126);
        assert(k >= 33 && k <= 126);
    }

    long h = strtol("B",NULL,16);

    char name[NAME_MAX] = {0};
    realpath("/dev/disk/by-uuid/8d110083-0761-4887-8106-cea62b375936",name);
    printf("%s\n",name);
    
    char src1[] = {"测试test测试知道"};
    uint8_t dst1[100] = {0};
    uint8_t dst2[100] = {0};
    ssize_t r1 = abcdk_iconv2("UTF-8","UCS-4",  src1, strlen(src1), dst1, 100, NULL);

    int r3 = abcdk_cslen(dst1,4);
    ssize_t r = abcdk_iconv2("UCS-4", "UTF-8", dst1, r1, dst2, 100, NULL);

    abcdk_sockaddr_t dd = {0};

    abcdk_sockaddr_from_string(&dd,"127.0.0.1:17007",1);

#endif

#ifdef HAVE_OPENSSL

    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();

#endif //HAVE_OPENSSL

    ABCDK_ASSERT(abcdk_option_count(args, "--func") > 0, "未指定命令。");

    if(abcdk_strcmp(func,"test_mux",0)==0)
        test_mux(args);

    if(abcdk_strcmp(func,"test_ffmpeg",0)==0)
        test_ffmpeg(args);

    if(abcdk_strcmp(func,"test_bmp",0)==0)
        test_bmp(args);

    if(abcdk_strcmp(func,"test_freeimage",0)==0)
        test_freeimage(args);

    if(abcdk_strcmp(func,"test_uri",0)==0)
        test_uri(args);

    if (abcdk_strcmp(func, "test_strrep", 0) == 0)
        test_strrep(args);

    if (abcdk_strcmp(func, "test_html", 0) == 0)
        test_html(args);

    if (abcdk_strcmp(func, "test_fnmatch", 0) == 0)
        test_fnmatch(args);

    if (abcdk_strcmp(func, "test_crc32", 0) == 0)
        test_crc32(args);

    if (abcdk_strcmp(func, "test_robots", 0) == 0)
        test_robots(args);

    if (abcdk_strcmp(func, "test_fuse", 0) == 0)
        test_fuse(args);

    if (abcdk_strcmp(func, "test_mp4", 0) == 0)
        test_mp4(args);

    if (abcdk_strcmp(func, "test_dirent", 0) == 0)
        test_dirent(args);
    
    if (abcdk_strcmp(func, "test_netlink", 0) == 0)
        test_netlink(args);

    if (abcdk_strcmp(func, "test_iwscan", 0) == 0)
        test_iwscan(args);

    if (abcdk_strcmp(func, "test_hexdump", 0) == 0)
        test_hexdump(args);

    if (abcdk_strcmp(func, "test_video", 0) == 0)
        test_video(args);

    if (abcdk_strcmp(func, "test_com", 0) == 0)
        test_com(args);

    if (abcdk_strcmp(func, "test_mpi", 0) == 0)
        test_mpi(args);

    if (abcdk_strcmp(func, "test_lz4", 0) == 0)
        test_lz4(args);

    if (abcdk_strcmp(func, "test_archive", 0) == 0)
        test_archive(args);

    if (abcdk_strcmp(func, "test_modbus", 0) == 0)
        test_modbus(args);

    if (abcdk_strcmp(func, "test_libusb", 0) == 0)
        test_libusb(args);

    if (abcdk_strcmp(func, "test_openssl", 0) == 0)
        test_openssl(args);

    if (abcdk_strcmp(func, "test_mqtt", 0) == 0)
       test_mqtt(args);

    if (abcdk_strcmp(func, "test_http", 0) == 0)
       test_http(args);

    if (abcdk_strcmp(func, "test_redis", 0) == 0)
       test_redis(args);
    
    if (abcdk_strcmp(func, "test_cert_verify", 0) == 0)
       test_cert_verify(args);
    
    if (abcdk_strcmp(func, "test_json", 0) == 0)
       test_json(args);
    
    if (abcdk_strcmp(func, "test_refer_count", 0) == 0)
       test_refer_count(args);
    
    if (abcdk_strcmp(func, "test_comm", 0) == 0)
       test_comm(args);
        
    if (abcdk_strcmp(func, "test_easy", 0) == 0)
       test_easy(args);
    
    if (abcdk_strcmp(func, "test_blkid", 0) == 0)
        test_blkid(args);

    if (abcdk_strcmp(func, "test_bloom", 0) == 0)
        test_bloom(args);

    if (abcdk_strcmp(func, "test_basecode", 0) == 0)
        test_basecode(args);
    
    if (abcdk_strcmp(func, "test_setns", 0) == 0)
        test_setns(args);
    
    if (abcdk_strcmp(func, "test_notify", 0) == 0)
        test_notify(args);
    
    if (abcdk_strcmp(func, "test_scsi", 0) == 0)
        test_scsi(args);
    
    if (abcdk_strcmp(func, "test_fcgi", 0) == 0)
        test_fcgi(args);

    if (abcdk_strcmp(func, "test_geom", 0) == 0)
        test_geom(args);

    if (abcdk_strcmp(func, "test_fb", 0) == 0)
        test_fb(args);

    if (abcdk_strcmp(func, "test_udev", 0) == 0)
        test_udev(args);

    if (abcdk_strcmp(func, "test_dmtx", 0) == 0)
        test_dmtx(args);

    if (abcdk_strcmp(func, "test_zbar", 0) == 0)
        test_zbar(args);
    
    if (abcdk_strcmp(func, "test_ndarray", 0) == 0)
        test_ndarray(args);

    if (abcdk_strcmp(func, "test_unix_sock", 0) == 0)
        test_unix_sock(args);

    if (abcdk_strcmp(func, "test_mtab", 0) == 0)
        test_mtab(args);

    if (abcdk_strcmp(func, "test_block", 0) == 0)
        test_block(args);

    if (abcdk_strcmp(func, "test_sqlite", 0) == 0)
        test_sqlite(args);

    if (abcdk_strcmp(func, "test_reader", 0) == 0)
        test_reader(args);

    if (abcdk_strcmp(func, "test_kafka", 0) == 0)
        test_kafka(args);

    if (abcdk_strcmp(func, "test_record", 0) == 0)
        test_record(args);

    if (abcdk_strcmp(func, "test_dup", 0) == 0)
        test_dup(args);

    if (abcdk_strcmp(func, "test_file_wholockme", 0) == 0)
        test_file_wholockme(args);

    if (abcdk_strcmp(func, "test_file_segment", 0) == 0)
        test_file_segment(args);

    if (abcdk_strcmp(func, "test_odbcpool", 0) == 0)
        test_odbcpool(args);

    if (abcdk_strcmp(func, "test_log", 0) == 0)
        test_log(args);

    abcdk_tree_free(&args);
    
    return 0;
}
