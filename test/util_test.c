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
#include <linux/serial.h>
#include "abcdk-util/general.h"
#include "abcdk-util/getargs.h"
#include "abcdk-util/geometry.h"
#include "abcdk-util/ffmpeg.h"
#include "abcdk-util/bmp.h"
#include "abcdk-util/freeimage.h"
#include "abcdk-util/uri.h"
#include "abcdk-util/html.h"
#include "abcdk-util/clock.h"
#include "abcdk-util/crc32.h"
#include "abcdk-util/robots.h"
#include "abcdk-util/dirent.h"
#include "abcdk-util/socket.h"
#include "abcdk-util/hexdump.h"
#include "abcdk-util/termios.h"
#include "abcdk-mp4/demuxer.h"
#include "abcdk-util/video.h"
#include "abcdk-auth/auth.h"
#include "abcdk-util/lz4.h"
#include "abcdk-util/openssl.h"
#include "abcdk-util/redis.h"
#include "abcdk-tls/tls.h"
#include "abcdk-util/json.h"

#ifdef HAVE_FUSE
#define FUSE_USE_VERSION 29
#include <fuse.h>
#endif //

#ifdef HAVE_LIBNM
#include <libnm/NetworkManager.h>
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


void test_log(abcdk_tree_t *args)
{
    abcdk_openlog(NULL,LOG_DEBUG,1);

    for(int i = LOG_EMERG ;i<= LOG_DEBUG;i++)
        syslog(i,"haha-%d",i);
}

void test_ffmpeg(abcdk_tree_t *args)
{

#ifdef HAVE_FFMPEG

    for(int i = 0;i<1000;i++)
    {
        enum AVPixelFormat pixfmt = (enum AVPixelFormat)i;

        int bits = abcdk_av_image_pixfmt_bits(pixfmt,0);
        int bits_pad = abcdk_av_image_pixfmt_bits(pixfmt,1);
        const char *name = abcdk_av_image_pixfmt_name(pixfmt);

        printf("%s(%d): %d/%d bits.\n",name,i,bits,bits_pad);
    }

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

    abcdk_resize_make(&r,width,height,dst_w,dst_h,0);

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

    int left = abcdk_resize_src2dst(&r,0,1);
    int top = abcdk_resize_src2dst(&r,0,0);
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
    const char *uri = abcdk_option_get(args,"--uri",0,"");

    abcdk_allocator_t * alloc = abcdk_uri_split(uri);
    assert(alloc);


    for(size_t i = 0;i<alloc->numbers;i++)
        printf("[%ld]: %s\n",i,alloc->pptrs[i]);

    abcdk_allocator_unref(&alloc);
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

    abcdk_allocator_t *t = abcdk_mmap2(name_p,0,0);
    if(!t)
        return;

    abcdk_buffer_t *buf = abcdk_buffer_alloc(t);
    if(!buf)
    {
        abcdk_allocator_unref(&t);
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
        
        abcdk_dirent_open(t,file);
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

    abcdk_allocator_t * k = abcdk_allocator_alloc(NULL,1,0);
    abcdk_allocator_t * p = abcdk_allocator_alloc(NULL,1,0);

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

  abcdk_allocator_t * scan_rsp = abcdk_allocator_alloc2(100000);

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

    abcdk_allocator_unref(&scan_rsp);
    abcdk_allocator_unref(&k);
    abcdk_allocator_unref(&p);

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

#endif //HAVE_LIBNM
#endif
}

void test_hexdump(abcdk_tree_t *args)
{
    const char *file_p = abcdk_option_get(args,"--file",0,"");

    abcdk_allocator_t * m = abcdk_mmap2(file_p,0,0);

    abcdk_hexdump_option_t opt = {0};

    if(abcdk_option_exist(args,"--show-addr"))
        opt.flag |= ABCDK_HEXDEMP_SHOW_ADDR;
    if(abcdk_option_exist(args,"--show-char"))
        opt.flag |= ABCDK_HEXDEMP_SHOW_CHAR;

    opt.width = abcdk_option_get_int(args,"--width",0,16);

    opt.keyword = abcdk_allocator_alloc(NULL,4,0);
    opt.palette = abcdk_allocator_alloc(NULL,3,0);

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

    abcdk_allocator_unref(&m);
    abcdk_allocator_unref(&opt.keyword);
    abcdk_allocator_unref(&opt.palette);
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

void test_auth(abcdk_tree_t *args)
{
    abcdk_tree_t *auth = abcdk_tree_alloc3(1);

    abcdk_auth_collect_dmi(auth);
    abcdk_auth_collect_mac(auth);
    abcdk_auth_add_valid_period2(auth,20,0);

 //   abcdk_option_fprintf(stderr,auth,NULL);

    assert(abcdk_auth_verify(auth)==0);

    abcdk_allocator_t *dump = abcdk_auth_serialize(auth);

    fprintf(stderr,"%s\n",dump->pptrs[0]);

    abcdk_allocator_t *ciphertext = abcdk_auth_encrypt(dump,ABCDK_AUTH_DEFAULT_KEY);
    abcdk_allocator_t *plaintext = abcdk_auth_decrypt(ciphertext,ABCDK_AUTH_DEFAULT_KEY);

    assert(memcmp(plaintext->pptrs[0],dump->pptrs[0],dump->sizes[0])==0);

    abcdk_auth_save2("/tmp/abcdk.auth",ciphertext->pptrs[0],ciphertext->sizes[0],ABCDK_AUTH_DEFAULT_MAGIC);
    abcdk_allocator_t *ciphertext2 = abcdk_auth_load2("/tmp/abcdk.auth",ABCDK_AUTH_DEFAULT_MAGIC);

    assert(memcmp(ciphertext->pptrs[0],ciphertext2->pptrs[0],ciphertext2->sizes[0])==0);
    abcdk_allocator_t *plaintext2 = abcdk_auth_decrypt(ciphertext2,ABCDK_AUTH_DEFAULT_KEY);

    assert(memcmp(plaintext2->pptrs[0],dump->pptrs[0],dump->sizes[0])==0);
    
    abcdk_allocator_unref(&dump);
    abcdk_allocator_unref(&ciphertext);
    abcdk_allocator_unref(&plaintext);
    abcdk_allocator_unref(&ciphertext2);
    abcdk_allocator_unref(&plaintext2);
    abcdk_tree_free(&auth);
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

    abcdk_allocator_t *s = abcdk_mmap2(src,0,0);

    size_t dsize = abcdk_endian_b_to_h32(ABCDK_PTR2U32(s->pptrs[0],0));

    abcdk_allocator_t *d = abcdk_allocator_alloc2(dsize);

    //LZ4_decompress_fast(s->pptrs[0]+4,d->pptrs[0],dsize);
    int m = abcdk_lz4_dec_fast(d->pptrs[0],dsize,s->pptrs[0]+4);

    abcdk_allocator_t *q = abcdk_allocator_alloc2(2000);

    int n = abcdk_lz4_enc_default(q->pptrs[0],q->sizes[0],d->pptrs[0],dsize);

    //assert(memcmp(q->pptrs[0],s->pptrs[0]+4,s->sizes[0]-4)==0);

    abcdk_allocator_t *p = abcdk_allocator_alloc2(dsize);

    int m2 = abcdk_lz4_dec_fast(p->pptrs[0],dsize,q->pptrs[0]);

    assert(memcmp(p->pptrs[0],d->pptrs[0],d->sizes[0])==0);

    abcdk_allocator_unref(&q);
    abcdk_allocator_unref(&p);

    int fd = abcdk_open(dst,1,0,1);
    ftruncate(fd,0);
    abcdk_write(fd,d->pptrs[0],dsize);
    abcdk_closep(&fd);

    abcdk_allocator_unref(&s);
    abcdk_allocator_unref(&d);


#endif 
}

void test_archive(abcdk_tree_t *args)
{
#ifdef HAVE_ARCHIVE

    const char *src = abcdk_option_get(args,"--src",0,"");
    const char *dst = abcdk_option_get(args,"--dst",0,"");

    struct archive *a = archive_write_new();
    struct archive_entry *entry = archive_entry_new();

  //  archive_write_add_filter_bzip2(a);
  //  archive_write_set_format_zip(a);

  //  archive_write_add_filter_gzip(a);
  //  archive_write_set_format_pax_restricted(a); // Note 1

    archive_write_set_format_gnutar(a);

    archive_write_open_filename(a, dst);

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

    char buf[500];

    for(;;)
    {
        ssize_t r = abcdk_read(fd,buf,500);
        if(r<=0)
            break;
        
        archive_write_data(a,buf,r);
    }

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
    addr.family = ABCDK_UNIX;
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

    addr.family = ABCDK_UNIX;
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

   SSL_library_init();
   OpenSSL_add_all_algorithms();
   SSL_load_error_strings();

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
        printf("%s %s\n", message->topic, (char*)message->payload);
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
        mosquitto_subscribe(mosq, NULL, "hello", 2);
    }
    else
    {
        fprintf(stderr, "Connect failed\n");
    }
}

void my_subscribe_callback(struct mosquitto *mosq, void *userdata, int mid, int qos_count, const int *granted_qos)
{
    int i;

    printf("Subscribed (mid: %d): %d", mid, granted_qos[0]);
    for (i = 1; i < qos_count; i++)
    {
        printf(", %d", granted_qos[i]);
    }
    printf("\n");
}

void my_log_callback(struct mosquitto *mosq, void *userdata, int level, const char *str)
{
    /* Pring all log messages regardless of level. */
    printf("%s\n", str);
}
#endif 

int test_mqtt(abcdk_tree_t *args)
{
#ifdef HAVE_MQTT
    int i;
    char *host = "localhost";
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
    int s = abcdk_socket(ABCDK_IPV4,0);

    abcdk_sockaddr_t a;
    a.family = ABCDK_IPV4;
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

    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();

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

void tls_event_cb(abcdk_tls_node *node, uint32_t event)
{
    abcdk_sockaddr_t addr;
    abcdk_tls_get_peername(node,&addr);
    char addr_str[100] = {0};
    abcdk_sockaddr_to_string(addr_str,&addr);

    switch(event)
    {
        case ABCDK_TLS_EVENT_CONNECT:
        {
            printf("Connected: %s\n",addr_str);

            abcdk_tls_set_timeout(node,5*1000);

            abcdk_tls_read_watch(node,0);
        }
        break;
        case ABCDK_TLS_EVENT_INPUT:
        {
            while(1)
            {
                char buf[100]={0};
                ssize_t r = abcdk_tls_read(node,buf,100);
                if(r<=0)
                    break;

                printf("%s",buf);
            }

            abcdk_tls_read_watch(node,1);
            abcdk_tls_write_watch(node);
            
        }
        break;
        case ABCDK_TLS_EVENT_OUTPUT:
        {
            abcdk_tls_write(node,"abcdk\n",6);
        }
        break;
        case ABCDK_TLS_EVENT_CLOSE:
        {
            printf("Disconnected: %s\n",addr_str);
        }
        break;
    }
}

void test_tls(abcdk_tree_t *args)
{

    signal(SIGPIPE,NULL);

    SSL_CTX *ssl_ctx = NULL;

#ifdef HAVE_OPENSSL

    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();

    const char *capath = abcdk_option_get(args,"--ca-path",0,NULL);

    if (capath)
    {
        ssl_ctx = abcdk_openssl_ssl_ctx_alloc(1, NULL, capath, 2);

        //X509_VERIFY_PARAM_set_purpose(param, X509_PURPOSE_ANY);

        abcdk_openssl_ssl_ctx_load_crt(ssl_ctx, abcdk_option_get(args, "--crt-file", 0, NULL),
                                       abcdk_option_get(args, "--key-file", 0, NULL),
                                       abcdk_option_get(args, "--key-pwd", 0, NULL));

        SSL_CTX_set_verify(ssl_ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
    }
#endif //HAVE_OPENSSL

    abcdk_sockaddr_t addr = {0};

    const char *listen_p = abcdk_option_get(args,"--listen",0,"0.0.0.0:12345");
    abcdk_sockaddr_from_string(&addr,listen_p,0);
   // abcdk_sockaddr_from_string(&addr,"[::]:12345",0);
   // addr.family = ABCDK_UNIX;
    //strcpy(addr.addr_un.sun_path,"/tmp/abcdk.txt2");

    assert(abcdk_tls_listen(&addr,ssl_ctx,NULL)==0);

    abcdk_sockaddr_t addr2 = {0};
  //  abcdk_sockaddr_from_string(&addr2,"www.baidu.com:80",1);

  //  assert(abcdk_tls_connect(&addr2,NULL,NULL)==0);

    #pragma omp parallel for num_threads(3)
    for (int i = 0; i < 3; i++)
    {
        abcdk_tls_loop(tls_event_cb);
    }

    abcdk_tls_cleanup();
}

void test_json(abcdk_tree_t *args)
{
#ifdef _json_h_

    const char *src = abcdk_option_get(args,"--src",0,NULL);

    json_object *src_obj = json_object_from_file(src);

    abcdk_json_readable(stdout,1,0,src_obj);

    abcdk_json_unref(&src_obj);

#endif //_json_h_
}

void test_refer_count(abcdk_tree_t *args)
{
    int user = abcdk_option_get_int(args,"--user",0,10);

    abcdk_allocator_t * p= abcdk_allocator_alloc2(100);

#pragma omp parallel for num_threads(user)
    for (int i = 0; i < 100000; i++)
    {
        abcdk_allocator_t *q = abcdk_allocator_refer(p);

        usleep(10*1000);

        abcdk_allocator_unref(&q);
    }

    abcdk_allocator_unref(&p);
}

int main(int argc, char **argv)
{
    abcdk_openlog(NULL,LOG_DEBUG,1);

    abcdk_tree_t *args = abcdk_tree_alloc3(1);

    abcdk_getargs(args,argc,argv,"--");
    
    abcdk_option_fprintf(stderr,args,NULL);

    const char *func = abcdk_option_get(args,"--func",0,"");

   // abcdk_clock_reset();

    int a = 0x112233;
    int b = 0;
    char a8[3] = {0};

    abcdk_endian_h_to_b24(a8,a);
    b = abcdk_endian_b_to_h24(a8);
    assert(a == b);

    abcdk_endian_h_to_l24(a8,a);
    b = abcdk_endian_l_to_h24(a8);
    assert(a == b);
    

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

    if (abcdk_strcmp(func, "test_auth", 0) == 0)
        test_auth(args);

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
        
    if (abcdk_strcmp(func, "test_tls", 0) == 0)
       test_tls(args);
    
    if (abcdk_strcmp(func, "test_json", 0) == 0)
       test_json(args);
    
    if (abcdk_strcmp(func, "test_refer_count", 0) == 0)
       test_refer_count(args);

    abcdk_tree_free(&args);
    
    return 0;
}
