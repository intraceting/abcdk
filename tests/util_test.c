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
#include "abcdkutil/general.h"
#include "abcdkutil/getargs.h"
#include "abcdkutil/geometry.h"
#include "abcdkutil/ffmpeg.h"
#include "abcdkutil/bmp.h"
#include "abcdkutil/freeimage.h"
#include "abcdkutil/uri.h"
#include "abcdkutil/html.h"
#include "abcdkutil/clock.h"
#include "abcdkutil/crc32.h"
#include "abcdkutil/robots.h"

#ifdef HAVE_FUSE
#define FUSE_USE_VERSION 29
#include <fuse.h>
#endif //


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
    char str[]={"abcd?*Qcde"};
    char wd[]={"abc?\\?\\*q*e"};

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

static int _test_robots_dump_cb(size_t deep, abcdk_tree_t *node, void *opaque)
{
    if (deep == 0)
    {
        abcdk_tree_fprintf(stderr, deep, node, "%s\n", ".");
    }
    else
    {
        abcdk_tree_fprintf(stderr, deep, node, "%hhu: %s\n",
                           ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_ROBOTS_FLAG], 0),
                           ABCDK_PTR2I8PTR(node->alloc->pptrs[ABCDK_ROBOTS_PATH], 0));
    }

    return 1;
}

void test_robots(abcdk_tree_t *args)
{
    const char *file = abcdk_option_get(args,"--file",0,"");

    abcdk_tree_t *t = abcdk_robots_parse_file(file,"Baiduspider");

    abcdk_tree_iterator_t it = {0,_test_robots_dump_cb,NULL};

    abcdk_tree_scan(t,&it);


    abcdk_tree_free(&t);
}

#ifdef _FUSE_H_

#define MP4_PATH "/home/user/下载/"

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

void test_fmp4(abcdk_tree_t *args)
{
    const char *name_p = abcdk_option_get(args,"--file",0,"");

    abcdk_allocator_t *t = abcdk_mmap2(name_p,0,0);
    if(!t)
        return;

    void *p = t->pptrs[0];

    uint32_t size = abcdk_endian_b_to_h32(ABCDK_PTR2U32(p, 0));
    uint32_t type = ABCDK_PTR2U32(p, 4);
    for (int i = 0; i < 4; i++)
        printf("%c", ABCDK_PTR2I8(&type, i));
    printf("\n");
    uint32_t mb = ABCDK_PTR2U32(p, 8);
    for (int i = 0; i < 4; i++)
        printf("%c", ABCDK_PTR2I8(&mb, i));
    printf("\n");
    uint32_t mv = abcdk_endian_b_to_h32(ABCDK_PTR2U32(p, 12));
        printf("%u\n",mv);

    uint32_t size2 = 16;
    while(size2<size)
    {
        uint32_t cb = ABCDK_PTR2U32(p, size2);
        for (int i = 0; i < 4; i++)
            printf("%c", ABCDK_PTR2I8(&cb, i));
        printf("\n");

        size2+=4;
    }
   

    abcdk_allocator_unref(&t);
}


int main(int argc, char **argv)
{
    abcdk_openlog(NULL,LOG_DEBUG,1);

    abcdk_tree_t *args = abcdk_tree_alloc3(1);

    abcdk_getargs(args,argc,argv,"--");
    
    abcdk_option_fprintf(stderr,args,NULL);

    const char *func = abcdk_option_get(args,"--func",0,"");

    abcdk_clock_reset();

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

    if (abcdk_strcmp(func, "test_fmp4", 0) == 0)
        test_fmp4(args);

    abcdk_tree_free(&args);
    
    return 0;
}