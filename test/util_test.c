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
#include "abcdk-mp4/demuxer.h"

#ifdef HAVE_FUSE
#define FUSE_USE_VERSION 29
#include <fuse.h>
#endif //

#ifdef HAVE_NETWORKMANAGER
#include <libnm/NetworkManager.h>
#endif 

void test_log(abcdk_tree_t *args)
{
    abcdk_openlog(NULL,LOG_DEBUG,1);

    for(int i = LOG_EMERG ;i<= LOG_DEBUG;i++)
        syslog(i,"haha-%d",i);
}

void test_ffmpeg(abcdk_tree_t *args)
{

#if defined(AVUTIL_AVUTIL_H) && defined(SWSCALE_SWSCALE_H)

    for(int i = 0;i<1000;i++)
    {
        enum AVPixelFormat pixfmt = (enum AVPixelFormat)i;

        if(!ABCDK_AVPIXFMT_CHECK(pixfmt))
            continue;

        int bits = abcdk_av_image_pixfmt_bits(pixfmt,0);
        int bits_pad = abcdk_av_image_pixfmt_bits(pixfmt,1);
        const char *name = abcdk_av_image_pixfmt_name(pixfmt);

        printf("%s(%d): %d/%d bits.\n",name,i,bits,bits_pad);
    }

    
    abcdk_av_image_t src = {AV_PIX_FMT_YUV420P,{NULL,NULL,NULL,NULL},{0,0,0,0},1920,1080};
    abcdk_av_image_t dst = {AV_PIX_FMT_YUV420P,{NULL,NULL,NULL,NULL},{0,0,0,0},1920,1080};
    abcdk_av_image_t dst2 = {AV_PIX_FMT_BGR32,{NULL,NULL,NULL,NULL},{0,0,0,0},1920,1080};

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

#endif //AVUTIL_AVUTIL_H && SWSCALE_SWSCALE_H




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

    abcdk_clock_dot(NULL);

    abcdk_tree_t *t = abcdk_html_parse_file(file);

    printf("%lu\n",abcdk_clock_step(NULL));

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

void _mp4_atom_destroy_cb(abcdk_allocator_t *alloc, void *opaque)
{
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)alloc->pptrs[0];

    abcdk_allocator_unref(&atom->cont);
}

void _mp4_dump_ftyp(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_ftyp_t *cont = (abcdk_mp4_atom_ftyp_t *)atom->cont->pptrs[0];
    

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

    abcdk_allocator_unref(&cont->compat);
}

void _mp4_dump_mvhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_mvhd_t *cont = (abcdk_mp4_atom_mvhd_t *)atom->cont->pptrs[0];

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

void _mp4_dump_udta(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_udta_t *cont = (abcdk_mp4_atom_udta_t *)atom->cont->pptrs[0];

    abcdk_tree_t *p = abcdk_tree_child(cont->entries,1);
    while (p)
    {
        abcdk_mp4_atom_t *q = (abcdk_mp4_atom_t *)p->alloc->pptrs[0];
        fprintf(stdout, "[offset=%lu,size=%lu,type=%c%c%c%c]",
                           q->off_head, q->size, q->type.u8[0], q->type.u8[1], q->type.u8[2], q->type.u8[3]);
        
        p = abcdk_tree_sibling(p,0);
    }
    
    abcdk_tree_free(&cont->entries);
}

void _mp4_dump_tkhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_tkhd_t *cont = (abcdk_mp4_atom_tkhd_t *)atom->cont->pptrs[0];

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
    abcdk_mp4_atom_mdhd_t *cont = (abcdk_mp4_atom_mdhd_t *)atom->cont->pptrs[0];

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
    abcdk_mp4_atom_hdlr_t *cont = (abcdk_mp4_atom_hdlr_t *)atom->cont->pptrs[0];


    fprintf(stdout,"type=%c%c%c%c, ",
                           cont->type.u8[0], cont->type.u8[1], cont->type.u8[2], cont->type.u8[3]);

    fprintf(stdout,"subtype=%c%c%c%c, ",
                           cont->subtype.u8[0], cont->subtype.u8[1], cont->subtype.u8[2], cont->subtype.u8[3]);

    if(cont->name)
        fprintf(stdout,"name='%s' ",cont->name->pptrs[0]);

    abcdk_allocator_unref(&cont->name);
}


void _mp4_dump_vmhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_vmhd_t *cont = (abcdk_mp4_atom_vmhd_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "mode=%u,",cont->mode);
    fprintf(stdout, "opcolor=%hu,%hu,%hu",cont->opcolor[0],cont->opcolor[1],cont->opcolor[2]);
}

void _mp4_dump_dref(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_dref_t *cont = (abcdk_mp4_atom_dref_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    abcdk_tree_t *p = abcdk_tree_child(cont->entries,1);
    while (p)
    {
        abcdk_mp4_atom_t *q = (abcdk_mp4_atom_t *)p->alloc->pptrs[0];
        fprintf(stdout, "[offset=%lu,size=%lu,type=%c%c%c%c]",
                           q->off_head, q->size, q->type.u8[0], q->type.u8[1], q->type.u8[2], q->type.u8[3]);
        
        p = abcdk_tree_sibling(p,0);
    }
    
    abcdk_tree_free(&cont->entries);
}

void _mp4_dump_stsd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stsd_t *cont = (abcdk_mp4_atom_stsd_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->entries)
        return ;

    abcdk_tree_t *p = abcdk_tree_child(cont->entries,1);
    while (p)
    {
        abcdk_mp4_atom_t *q = (abcdk_mp4_atom_t *)p->alloc->pptrs[0];
        fprintf(stdout, "[offset=%lu,size=%lu,type=%c%c%c%c]",
                           q->off_head, q->size, q->type.u8[0], q->type.u8[1], q->type.u8[2], q->type.u8[3]);
        
        p = abcdk_tree_sibling(p,0);
    }
    
    abcdk_tree_free(&cont->entries);
}

void _mp4_dump_stts(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stts_t *cont = (abcdk_mp4_atom_stts_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers;i++)
    {
        fprintf(stdout, "count=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));
        fprintf(stdout, "duration=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],4));
    }

    
    abcdk_allocator_unref(&cont->tables);
}

void _mp4_dump_ctts(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_ctts_t *cont = (abcdk_mp4_atom_ctts_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "count=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));
        fprintf(stdout, "offset=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],4));
    }

    
    abcdk_allocator_unref(&cont->tables);
}

void _mp4_dump_stsc(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stsc_t *cont = (abcdk_mp4_atom_stsc_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "Firstchunk=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));
        fprintf(stdout, "perchunk=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],4));
        fprintf(stdout, "ID=%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],8));
    }

    
    abcdk_allocator_unref(&cont->tables);
}

void _mp4_dump_stsz(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stsz_t *cont = (abcdk_mp4_atom_stsz_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "[samplesize=%u],",cont->samplesize);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));

    }

    
    abcdk_allocator_unref(&cont->tables);
}


void _mp4_dump_stco(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stco_t *cont = (abcdk_mp4_atom_stco_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i<10;i++)
    {
        fprintf(stdout, "%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));

    }

    
    abcdk_allocator_unref(&cont->tables);
}

void _mp4_dump_stss(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_stss_t *cont = (abcdk_mp4_atom_stss_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    if(!cont->tables)
        return;

    for(size_t i= 0 ;i<cont->tables->numbers && i < 10;i++)
    {
        fprintf(stdout, "%u,",ABCDK_PTR2U32(cont->tables->pptrs[i],0));

    }

    
    abcdk_allocator_unref(&cont->tables);
}


void _mp4_dump_gmhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_gmhd_t *cont = (abcdk_mp4_atom_gmhd_t *)atom->cont->pptrs[0];

    abcdk_tree_t *p = abcdk_tree_child(cont->entries,1);
    while (p)
    {
        abcdk_mp4_atom_t *q = (abcdk_mp4_atom_t *)p->alloc->pptrs[0];
        fprintf(stdout, "[offset=%lu,size=%lu,type=%c%c%c%c]",
                           q->off_head, q->size, q->type.u8[0], q->type.u8[1], q->type.u8[2], q->type.u8[3]);
        
        p = abcdk_tree_sibling(p,0);
    }
    
    abcdk_tree_free(&cont->entries);
}

void _mp4_dump_smhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_smhd_t *cont = (abcdk_mp4_atom_smhd_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "balance=%hhu.%hhu",cont->balance>>8,cont->balance&0xff);
}

void _mp4_dump_elst(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_elst_t *cont = (abcdk_mp4_atom_elst_t *)atom->cont->pptrs[0];

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

    
    abcdk_allocator_unref(&cont->tables);
}

void _mp4_dump_mehd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_mehd_t *cont = (abcdk_mp4_atom_mehd_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "duration=%lu",cont->duration);
}

void _mp4_dump_trex(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_trex_t *cont = (abcdk_mp4_atom_trex_t *)atom->cont->pptrs[0];

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
    abcdk_mp4_atom_mfhd_t *cont = (abcdk_mp4_atom_mfhd_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "sn=%lu,",cont->sn);

}

void _mp4_dump_tfhd(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_tfhd_t *cont = (abcdk_mp4_atom_tfhd_t *)atom->cont->pptrs[0];

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
    abcdk_mp4_atom_tfdt_t *cont = (abcdk_mp4_atom_tfdt_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "time=%lu,",cont->base_decode_time);

}

void _mp4_dump_trun(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_trun_t *cont = (abcdk_mp4_atom_trun_t *)atom->cont->pptrs[0];

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

    
    abcdk_allocator_unref(&cont->tables);
}

void _mp4_dump_mfro(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_mfro_t *cont = (abcdk_mp4_atom_mfro_t *)atom->cont->pptrs[0];

    fprintf(stdout, "version=%hhu,flag=[%08x],",cont->version,cont->flags);

    fprintf(stdout, "size=%lu,",cont->size);

}


void _mp4_dump_tfra(size_t deep, abcdk_tree_t *node, void *opaque)
{
    int fd = (int64_t)opaque;
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)node->alloc->pptrs[0];
    abcdk_mp4_atom_tfra_t *cont = (abcdk_mp4_atom_tfra_t *)atom->cont->pptrs[0];

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

    
    abcdk_allocator_unref(&cont->tables);
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

        abcdk_mp4_read_content(fd,atom);

        if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_FTYP)
            _mp4_dump_ftyp(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MVHD)
            _mp4_dump_mvhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_UDTA)
            _mp4_dump_udta(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_TKHD)
            _mp4_dump_tkhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_MDHD)
            _mp4_dump_mdhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_HDLR)
            _mp4_dump_hdlr(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_VMHD)
            _mp4_dump_vmhd(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_DREF)
            _mp4_dump_dref(deep, node, opaque);
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_STSD)
            _mp4_dump_stsd(deep, node, opaque);
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
        else if (atom->type.u32 == ABCDK_MP4_ATOM_TYPE_GMHD)
            _mp4_dump_gmhd(deep, node, opaque);
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

void show_mp4_info(int fd)
{
    abcdk_tree_t *root = abcdk_mp4_read_probe2(fd,0,-1UL, ABCDK_MP4_ATOM_TYPE_MOOV);

    abcdk_tree_t *f = abcdk_mp4_find2(root,ABCDK_MP4_ATOM_TYPE_STCO);
    abcdk_mp4_atom_t *atom = (abcdk_mp4_atom_t *)f->alloc->pptrs[0];

    abcdk_mp4_read_content(fd,atom);

    _mp4_dump_stco(0, f, NULL);

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

#if 1

    abcdk_tree_t *root = abcdk_mp4_read_probe(fd,0,-1UL, NULL);


    abcdk_tree_iterator_t it = {0,mp4_dump_cb,(void*)(int64_t)fd};
    abcdk_tree_scan(root,&it);

    printf("\natoms:%d\n",atoms);


    abcdk_tree_free(&root);
#else 

    show_mp4_info(fd);

#endif 

    abcdk_closep(&fd);

#endif 
}

int dirent_dump_cb(size_t deep, abcdk_tree_t *node, void *opaque)
{

    if(deep == -1)
        return -1;

    if(deep >1)
        return 0;

    char *path = (char*)(node->alloc->pptrs[ABCDK_DIRENT_NAME]);
    struct stat *stat = (struct stat *)(node->alloc->pptrs[ABCDK_DIRENT_STAT]);


    char name[NAME_MAX] ={0};
    abcdk_basename(name,path);

#if 1
    abcdk_tree_fprintf(stderr,deep,node,"[%lu, %lu, %lu] %s%s\n",
                        ABCDK_PTR2PTR(abcdk_dirent_counter_t,node->alloc->pptrs[ABCDK_DIRENT_DIRS], 0)->nums,
                        ABCDK_PTR2PTR(abcdk_dirent_counter_t,node->alloc->pptrs[ABCDK_DIRENT_REGS], 0)->nums,
                        ABCDK_PTR2PTR(abcdk_dirent_counter_t,node->alloc->pptrs[ABCDK_DIRENT_REGS], 0)->sizes,
                        name,(S_ISDIR(stat->st_mode)?"/":""));
#else 
    abcdk_tree_fprintf(stderr,0,deep,node,"%s(%s)\n",name,path);
#endif
    return 1;
}

int dirent_match_cb(size_t depth,abcdk_tree_t *node,void *opaque)
{
    abcdk_tree_t *args = (abcdk_tree_t *)opaque;
    const char *wildcard = abcdk_option_get(args,"--wildcard",0,NULL);

    char *name = (char*)(node->alloc->pptrs[ABCDK_DIRENT_NAME]);
    struct stat *stat = (struct stat *)(node->alloc->pptrs[ABCDK_DIRENT_STAT]);

    if(!wildcard)
        return 0;

    if(S_ISDIR(stat->st_mode))
        return 0;

    int chk = abcdk_fnmatch(name,wildcard,0,0);

    if(chk == 0)
        return 0;

    return -1;
}

void test_dirent(abcdk_tree_t *args)
{
    const char *path_p = abcdk_option_get(args,"--path",0,"");


    abcdk_dirent_filter_t f = {dirent_match_cb,args};
    abcdk_tree_t * t = abcdk_dirent_scan(path_p,&f);


    abcdk_tree_iterator_t it = {0,dirent_dump_cb,NULL};
    abcdk_tree_scan(t,&it);

    abcdk_tree_free(&t);
    
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

#ifdef HAVE_NETWORKMANAGER
static void
request_rescan_cb (GObject *object, GAsyncResult *result, gpointer user_data)
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

#endif //#ifdef HAVE_NETWORKMANAGER

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

#ifdef HAVE_NETWORKMANAGER
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

#endif //#ifdef HAVE_NETWORKMANAGER
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

int main(int argc, char **argv)
{
    abcdk_openlog(NULL,LOG_DEBUG,1);

    abcdk_tree_t *args = abcdk_tree_alloc3(1);

    abcdk_getargs(args,argc,argv,"--");
    
    abcdk_option_fprintf(stderr,args,NULL);

    const char *func = abcdk_option_get(args,"--func",0,"");

    abcdk_clock_reset();

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

    abcdk_tree_free(&args);
    
    return 0;
}