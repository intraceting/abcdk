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
#include "abcdk-util/general.h"
#include "abcdk-util/getargs.h"
#include "abcdk-mp4/demuxer.h"

typedef struct _abcdkm4j_ctx
{
    int errcode;

    abcdk_tree_t *args;
    
    const char *file;
    const char *save;

    char in_name[NAME_MAX];

    int in_fd;
    int out_fd[16];

    abcdk_tree_t *doc;

    abcdk_tree_t *moov_p;
    abcdk_tree_t *mvex_p;
    abcdk_tree_t *trak_p;
 
    abcdk_tree_t *tkhd_p;
    abcdk_tree_t *hdlr_p;
    abcdk_tree_t *stsz_p;
    abcdk_tree_t *stss_p;
    abcdk_tree_t *stts_p;
    abcdk_tree_t *ctts_p;
    abcdk_tree_t *stsc_p;
    abcdk_tree_t *stco_p;
    abcdk_tree_t *avc1_p;
    abcdk_tree_t *avcc_p;
    abcdk_tree_t *mp4a_p;
    abcdk_tree_t *esds_p;

    abcdk_mp4_atom_t *tkhd;
    abcdk_mp4_atom_t *hdlr;
    abcdk_mp4_atom_t *stsz;
    abcdk_mp4_atom_t *stss;
    abcdk_mp4_atom_t *stts;
    abcdk_mp4_atom_t *ctts;
    abcdk_mp4_atom_t *stco;
    abcdk_mp4_atom_t *stsc;
    abcdk_mp4_atom_t *avc1;
    abcdk_mp4_atom_t *avcc;
    abcdk_mp4_atom_t *mp4a;
    abcdk_mp4_atom_t *esds;

    struct
    {
        int write_adts;
        int objecttype;
        int sample_rate_index;
        int channel_conf;
    } adts_ctx;

}abcdkm4j_ctx;

void _abcdkm4j_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    abcdk_proc_basename(name);

    fprintf(stderr, "\n%s 构建 %s\n", name, BUILD_TIME);
    fprintf(stderr, "\n%s 版本 %d.%d-%d\n", name, VERSION_MAJOR, VERSION_MINOR, VERSION_RELEASE);

    if (only_version)
        ABCDK_ERRNO_AND_RETURN0(0);

    fprintf(stderr, "\n摘要:\n");

    fprintf(stderr, "\n%s [ --file < FILE > ] [ OPTIONS ] \n", name);

    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\tMP4视音频提取器，仅支持H264和ACC。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\t显示版本信息。\n");

    fprintf(stderr, "\n\t--file < FILE >\n");
    fprintf(stderr, "\t\t文件(包括路径)。\n");

    fprintf(stderr, "\n\t--save < PATH >\n");
    fprintf(stderr, "\t\t保存路径。默认：./\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

/*代码来源于网络，稍有修改。*/
int _abcdkm4j_aac_decode_extradata(abcdkm4j_ctx *ctx, unsigned char *pbuf, int bufsize)
{
    int aot, aotext, samfreindex;
    int i, channelconfig;
    unsigned char *p = pbuf;

    assert(bufsize >= 2);

    aot = (p[0] >> 3) & 0x1f;

    if (aot == 31)
    {
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
    
    ctx->adts_ctx.objecttype = aot-1;
    ctx->adts_ctx.sample_rate_index = samfreindex;
    ctx->adts_ctx.channel_conf = channelconfig;
    ctx->adts_ctx.write_adts = 1;

    return 0;
}

/*代码来源于网络，稍有修改。*/
int _abcdkm4j_aac_set_adts_head(abcdkm4j_ctx *ctx, unsigned char *buf, int size)
{
#define ABCDKM4J_ADTS_HEADER_SIZE   7

    unsigned char byte;
    if (size < ABCDKM4J_ADTS_HEADER_SIZE)
        return -1;

    buf[0] = 0xff;

    buf[1] = 0xf1;
    byte = 0;
    byte |= (ctx->adts_ctx.objecttype & 0x03) << 6;
    byte |= (ctx->adts_ctx.sample_rate_index & 0x0f) << 2;
    byte |= (ctx->adts_ctx.channel_conf & 0x07) >> 2;

    buf[2] = byte;
    byte = 0;
    byte |= (ctx->adts_ctx.channel_conf & 0x07) << 6;
    byte |= (ABCDKM4J_ADTS_HEADER_SIZE + size) >> 11;

    buf[3] = byte;
    byte = 0;
    byte |= (ABCDKM4J_ADTS_HEADER_SIZE + size) >> 3;

    buf[4] = byte;
    byte = 0;
    byte |= ((ABCDKM4J_ADTS_HEADER_SIZE + size) & 0x7) << 5;
    byte |= (0x7ff >> 6) & 0x1f;

    buf[5] = byte;
    byte = 0;
    byte |= (0x7ff & 0x3f) << 2;

    buf[6] = byte;

    return 0;
}

void _abcdkm4j_fmp4_dump(abcdkm4j_ctx *ctx)
{
    
}

void _abcdkm4j_dump_video(abcdkm4j_ctx *ctx)
{
    size_t buf_size = 4*8192*8192;
    char *buf = NULL;
    char *out_file = NULL;
    abcdk_mp4_tag_t sc4;//StartCode(4 Bytes)

    ctx->avc1_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_AVC1, 1, 1);
    ctx->avcc_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_AVCC, 1, 1);

    ctx->avc1 = (abcdk_mp4_atom_t*)(ctx->avc1_p?ctx->avc1_p->alloc->pptrs[0]:NULL);
    ctx->avcc = (abcdk_mp4_atom_t*)(ctx->avcc_p?ctx->avcc_p->alloc->pptrs[0]:NULL);

    if(!ctx->avc1)
    {
        syslog(LOG_ERR, "仅支持H264编码提取。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }

    sc4.u32 = ABCDK_MP4_ATOM_MKTAG('\0','\0','\0','\1');

    out_file = abcdk_heap_alloc(PATH_MAX);
    if(!out_file)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    buf = abcdk_heap_alloc(buf_size);
    if(!buf)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    memset(out_file,0,PATH_MAX);
    sprintf(out_file,"%s/%s-%u.h264",ctx->save,ctx->in_name,ctx->tkhd->data.tkhd.trackid);

    ctx->out_fd[0] = abcdk_open(out_file, 1, 0, 1);
    if (ctx->out_fd[0] < 0)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    /*清空文件。*/
    ftruncate(ctx->out_fd[0],0);

#if 0
    abcdk_write(ctx->out_fd[0], ctx->avcc->data.avcc.extradata->pptrs[0],
                ctx->avcc->data.avcc.extradata->sizes[0]);
#endif

    for (size_t i = 1; i <= ctx->stsz->data.stsz.numbers; i++)
    {
        uint32_t chunk = 0, offset = 0, id = 0;
        abcdk_mp4_stsc_tell(&ctx->stsc->data.stsc, i, &chunk, &offset, &id);

        uint32_t offset2 = 0, size = 0;
        abcdk_mp4_stsz_tell(&ctx->stsz->data.stsz, offset, i, &offset2, &size);

        if (buf_size < size)
        {
            syslog(LOG_ERR, "仅支持长度小于%lu字节的数据帧。",buf_size);
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = ESPIPE, final);
        }

        lseek(ctx->in_fd, ctx->stco->data.stco.tables[chunk - 1].offset + offset2, SEEK_SET);

        abcdk_mp4_read(ctx->in_fd, buf, size);

        abcdk_write(ctx->out_fd[0], &sc4.u32, 4);
        abcdk_write(ctx->out_fd[0], ctx->avcc->data.avcc.sps->pptrs[0], ctx->avcc->data.avcc.sps->sizes[0]);
        abcdk_write(ctx->out_fd[0], &sc4.u32, 4);
        abcdk_write(ctx->out_fd[0], ctx->avcc->data.avcc.pps->pptrs[0], ctx->avcc->data.avcc.pps->sizes[0]);
        abcdk_write(ctx->out_fd[0], &sc4.u32, 4); //用起始码替换长度字段。
        abcdk_write(ctx->out_fd[0], buf + 4, size - 4);//跳过长度字段。
    }

final:

    abcdk_closep(&ctx->out_fd[0]);
    abcdk_heap_free(buf);
    abcdk_heap_free(out_file);
}


void _abcdkm4j_dump_audio(abcdkm4j_ctx *ctx)
{
    size_t buf_size = 1024*1024;
    char *buf = NULL;
    char *out_file = NULL;

    ctx->mp4a_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_MP4A, 1, 1);
    ctx->esds_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_ESDS, 1, 1);

    ctx->mp4a = (abcdk_mp4_atom_t*)(ctx->mp4a_p?ctx->mp4a_p->alloc->pptrs[0]:NULL);
    ctx->esds = (abcdk_mp4_atom_t*)(ctx->esds_p?ctx->esds_p->alloc->pptrs[0]:NULL);

    if(!ctx->mp4a)
    {
        syslog(LOG_ERR, "仅支持AAC编码提取。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }

    out_file = abcdk_heap_alloc(PATH_MAX);
    if(!out_file)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    buf = abcdk_heap_alloc(buf_size);
    if(!buf)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    memset(out_file,0,PATH_MAX);
    sprintf(out_file,"%s/%s-%u.aac",ctx->save,ctx->in_name,ctx->tkhd->data.tkhd.trackid);

    ctx->out_fd[0] = abcdk_open(out_file, 1, 0, 1);
    if (ctx->out_fd[0] < 0)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    /*清空文件。*/
    ftruncate(ctx->out_fd[0],0);

    /*解析ADTS信息。*/
    _abcdkm4j_aac_decode_extradata(ctx,ctx->esds->data.esds.dec_sp_info.extradata->pptrs[0],
                                   ctx->esds->data.esds.dec_sp_info.extradata->sizes[0]);

    for (size_t i = 1; i <= ctx->stsz->data.stsz.numbers; i++)
    {
        uint32_t chunk = 0, offset = 0, id = 0;
        abcdk_mp4_stsc_tell(&ctx->stsc->data.stsc, i, &chunk, &offset, &id);

        uint32_t offset2 = 0, size = 0;
        abcdk_mp4_stsz_tell(&ctx->stsz->data.stsz, offset, i, &offset2, &size);

        lseek(ctx->in_fd, ctx->stco->data.stco.tables[chunk - 1].offset + offset2, SEEK_SET);

        abcdk_mp4_read(ctx->in_fd, buf, size);

        char hdr[7] = {0};
        _abcdkm4j_aac_set_adts_head(ctx, hdr, size);

        abcdk_write(ctx->out_fd[0], hdr, 7);
        abcdk_write(ctx->out_fd[0], buf, size);
    }

final:

    abcdk_closep(&ctx->out_fd[0]);
    abcdk_heap_free(buf);
    abcdk_heap_free(out_file);
}


void _abcdkm4j_dump(abcdkm4j_ctx *ctx)
{
    /*一定要初始化，否则关闭时可能会出现异想不到的问题。*/
    for (int i = 0; i <= ABCDK_ARRAY_SIZE(ctx->out_fd); i++)
        ctx->out_fd[i] = -1;

    ctx->moov_p = abcdk_mp4_find2(ctx->doc,ABCDK_MP4_ATOM_TYPE_MOOV,1,1);
    if(!ctx->moov_p)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = ESPIPE, final);

    ctx->mvex_p = abcdk_mp4_find2(ctx->moov_p,ABCDK_MP4_ATOM_TYPE_MVEX,1,1);

    if(ctx->mvex_p)
    {
        _abcdkm4j_fmp4_dump(ctx);
        goto final;
    }

    for (int i = 0; i < ABCDK_ARRAY_SIZE(ctx->out_fd); i++)
    {
        ctx->trak_p = abcdk_mp4_find2(ctx->moov_p, ABCDK_MP4_ATOM_TYPE_TRAK, i + 1, 0);
        if (!ctx->trak_p)
            ABCDK_ERRNO_AND_GOTO1(0, final);

        ctx->tkhd_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_TKHD, 1, 1);
        ctx->hdlr_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_HDLR, 1, 1);
        ctx->stsz_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSZ, 1, 1);
        ctx->stss_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSS, 1, 1);
        ctx->stts_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STTS, 1, 1);
        ctx->ctts_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_CTTS, 1, 1);
        ctx->stsc_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSC, 1, 1);
        ctx->stco_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STCO, 1, 1);
        
        ctx->tkhd = (abcdk_mp4_atom_t*)(ctx->tkhd_p?ctx->tkhd_p->alloc->pptrs[0]:NULL);
        ctx->hdlr = (abcdk_mp4_atom_t*)(ctx->hdlr_p?ctx->hdlr_p->alloc->pptrs[0]:NULL);
        ctx->stsz = (abcdk_mp4_atom_t*)(ctx->stsz_p?ctx->stsz_p->alloc->pptrs[0]:NULL);
        ctx->stss = (abcdk_mp4_atom_t*)(ctx->stss_p?ctx->stss_p->alloc->pptrs[0]:NULL);
        ctx->stts = (abcdk_mp4_atom_t*)(ctx->stts_p?ctx->stts_p->alloc->pptrs[0]:NULL);
        ctx->ctts = (abcdk_mp4_atom_t*)(ctx->ctts_p?ctx->ctts_p->alloc->pptrs[0]:NULL);
        ctx->stco = (abcdk_mp4_atom_t*)(ctx->stco_p?ctx->stco_p->alloc->pptrs[0]:NULL);
        ctx->stsc = (abcdk_mp4_atom_t*)(ctx->stsc_p?ctx->stsc_p->alloc->pptrs[0]:NULL);

        if(ctx->hdlr->data.hdlr.subtype.u32 == ABCDK_MP4_ATOM_MKTAG('v', 'i', 'd', 'e'))
            _abcdkm4j_dump_video(ctx);
        else if(ctx->hdlr->data.hdlr.subtype.u32 == ABCDK_MP4_ATOM_MKTAG('s', 'o', 'u', 'n'))
            _abcdkm4j_dump_audio(ctx);
        
        /*有错误发生，提前终止。*/
        if(ctx->errcode)
            break;
    }

final:

    for (int i = 0; i <= ABCDK_ARRAY_SIZE(ctx->out_fd); i++)
        abcdk_closep(&ctx->out_fd[i]);
}

void _abcdkm4j_work(abcdkm4j_ctx *ctx)
{
    ctx->in_fd = -1;

    ctx->file = abcdk_option_get(ctx->args, "--file", 0, NULL);
    ctx->save = abcdk_option_get(ctx->args, "--save", 0, "./");

    if (!ctx->file || !*ctx->file)
    {
        syslog(LOG_ERR, "'--file FILE' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->file, R_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.", ctx->file, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    if (!ctx->save || !*ctx->save)
    {
        syslog(LOG_ERR, "'--save PATH' 不能省略，且不能为空。");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->save, W_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' %s.", ctx->save, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    ctx->in_fd = abcdk_open(ctx->file, 0, 0, 0);
    if (ctx->in_fd < 0)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    ctx->doc = abcdk_mp4_read_probe2(ctx->in_fd, 0, -1UL, 0);
    if (!ctx->doc)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    if(!abcdk_mp4_find2(ctx->doc,ABCDK_MP4_ATOM_TYPE_FTYP,1,1))
    {
        syslog(LOG_WARNING, "'%s' 可能不是MP4文件，或尚未支持此格式。", ctx->file);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }

    memset(ctx->in_name,0,sizeof(ctx->in_name));
    abcdk_basename(ctx->in_name,ctx->file);

    _abcdkm4j_dump(ctx);

final:

    abcdk_closep(&ctx->in_fd);
    abcdk_tree_free(&ctx->doc);
}

int main(int argc, char **argv)
{
    abcdkm4j_ctx ctx = {0};

    /*中文，UTF-8*/
    setlocale(LC_ALL, "zh_CN.UTF-8");

    ctx.args = abcdk_tree_alloc3(1);
    if (!ctx.args)
        goto final;

    abcdk_getargs(ctx.args, argc, argv, "--");

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdkm4j_print_usage(ctx.args, 0);
    }
    else if (abcdk_option_exist(ctx.args, "--version"))
    {
        _abcdkm4j_print_usage(ctx.args, 1);
    }
    else
    {
        _abcdkm4j_work(&ctx);
    }

final:
    
    abcdk_tree_free(&ctx.args);

    return ctx.errcode;
}