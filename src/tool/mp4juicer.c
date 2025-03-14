/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"

typedef struct _abcdk_m4j
{
    int errcode;

    abcdk_option_t *args;
    
    const char *file;
    const char *save;

    int ignore_video;
    int ignore_audio;
    int quiet;

    char in_name[NAME_MAX];
    size_t buf_size;
    char *buf;
    char *out_file;

    int in_fd;
    int out_fd;

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
    abcdk_tree_t *hev1_p;
    abcdk_tree_t *hvcc_p;
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
    abcdk_mp4_atom_t *hev1;
    abcdk_mp4_atom_t *hvcc;
    abcdk_mp4_atom_t *mp4a;
    abcdk_mp4_atom_t *esds;

    abcdk_tree_t *moof_p;
    abcdk_tree_t *mfhd_p;
    abcdk_tree_t *traf_p;
    abcdk_tree_t *tfhd_p;
    abcdk_tree_t *tfdt_p;
    abcdk_tree_t *trun_p;

    abcdk_mp4_atom_t *moof;
    abcdk_mp4_atom_t *mfhd;
    abcdk_mp4_atom_t *traf;
    abcdk_mp4_atom_t *tfhd;
    abcdk_mp4_atom_t *tfdt;
    abcdk_mp4_atom_t *trun;

    abcdk_aac_adts_header_t adts_hdr;

}abcdk_m4j_t;

void _abcdk_m4j_print_usage(abcdk_option_t *args, int only_version)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\tMP4视音频提取器，仅支持H264和ACC。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--file < FILE >\n");
    fprintf(stderr, "\t\t文件(包括路径)。\n");

    fprintf(stderr, "\n\t--save < PATH >\n");
    fprintf(stderr, "\t\t保存路径。默认：./\n");

    fprintf(stderr, "\n\t--ignore-video\n");
    fprintf(stderr, "\t\t忽略视频。默认：提取\n");

    fprintf(stderr, "\n\t--ignore-audio\n");
    fprintf(stderr, "\t\t忽略音频。默认：提取\n");

    ABCDK_ERRNO_AND_RETURN0(0);
}

int _abcdk_m4j_aac_decode_extradata(abcdk_m4j_t *ctx, uint8_t *data, int size)
{
    ctx->adts_hdr.profile = abcdk_bloom_read_number(data,size,0,5);
    if(ctx->adts_hdr.profile == 31)
    {
        ctx->adts_hdr.profile = 32 + abcdk_bloom_read_number(data,size,5,6);
        ctx->adts_hdr.sample_rate_index = abcdk_bloom_read_number(data,size,11,4);
        if(ctx->adts_hdr.sample_rate_index == 15)
            ctx->adts_hdr.channel_cfg = abcdk_bloom_read_number(data,size,15+24,4); //跳过24bits自定义的采样率。
        else
            ctx->adts_hdr.channel_cfg = abcdk_bloom_read_number(data,size,15,4); 
    }
    else
    {
        ctx->adts_hdr.sample_rate_index = abcdk_bloom_read_number(data,size,5,4);
        if(ctx->adts_hdr.sample_rate_index == 15)
            ctx->adts_hdr.channel_cfg = abcdk_bloom_read_number(data,size,9+24,4); //跳过24bits自定义的采样率。
        else
            ctx->adts_hdr.channel_cfg = abcdk_bloom_read_number(data,size,9,4); 
    }

    /*填充其它头部字段。*/
    ctx->adts_hdr.syncword = 0xfff;
    ctx->adts_hdr.id = 0;
    ctx->adts_hdr.protection_absent = 1;
    ctx->adts_hdr.adts_buffer_fullness = 0x7ff;

    return 0;
}

void _abcdk_m4j_dump_h264(abcdk_m4j_t *ctx)
{
    uint8_t sc[4];
    int sc_len;
    abcdk_h264_extradata_t exdata = {0};

    ctx->avcc_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_AVCC, 1, 1);

    if (!ctx->avcc_p)
    {
        fprintf(stderr, "H264描述信息不存在，忽略当前视频ID(%u)。\n", ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, final);
    }

    ctx->avc1 = (abcdk_mp4_atom_t *)(ctx->avc1_p->obj->pptrs[0]);
    ctx->avcc = (abcdk_mp4_atom_t *)(ctx->avcc_p->obj->pptrs[0]);

    abcdk_h264_extradata_deserialize(ctx->avcc->data.avcc.extradata->pptrs[0],ctx->avcc->data.avcc.extradata->sizes[0],&exdata);

    /*比真实长度少一个字节。*/
    sc_len = exdata.nal_length_size + 1;

    if (sc_len == 3)
        memcpy(sc, "\0\0\1", 3);
    else if (sc_len == 4)
        memcpy(sc, "\0\0\0\1", 4);
    else
    {
        fprintf(stderr, "H264仅支持001或0001格式起始码，忽略当前视频ID(%u)。\n", ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, final);
    }

    /*构造文件名。*/
    memset(ctx->out_file, 0, PATH_MAX);
    sprintf(ctx->out_file, "%s/%s-%u.h264", ctx->save, ctx->in_name, ctx->tkhd->data.tkhd.trackid);

    if (access(ctx->out_file, F_OK) == 0)
    {

        fprintf(stderr, "'%s' 已经存在，忽略当前视频ID(%u)。\n",ctx->out_file,ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, final);
    }

    fprintf(stdout, "%u: %s\n", ctx->tkhd->data.tkhd.trackid, ctx->out_file);

    ctx->out_fd = abcdk_open(ctx->out_file, 1, 0, 1);
    if (ctx->out_fd < 0)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    /*
     * 1：在流的头部写入SPS，PPS等。
     * 2：正常的做法是在每个关键帧前都写一次，但会增加流的体积。
    */
    abcdk_write(ctx->out_fd, sc, sc_len);
    abcdk_write(ctx->out_fd, exdata.sps->pptrs[0], exdata.sps->sizes[0]);
    abcdk_write(ctx->out_fd, sc, sc_len);
    abcdk_write(ctx->out_fd, exdata.pps->pptrs[0], exdata.pps->sizes[0]);

    if (ctx->mvex_p)
    {
        ctx->moof_p = abcdk_tree_child(ctx->doc, 1);
        while (ctx->moof_p)
        {
            ctx->moof = (abcdk_mp4_atom_t *)ctx->moof_p->obj->pptrs[0];
            if (ctx->moof->type.u32 != ABCDK_MP4_ATOM_TYPE_MOOF)
                goto moof_next;

            ctx->mfhd_p = abcdk_mp4_find2(ctx->moof_p, ABCDK_MP4_ATOM_TYPE_MFHD, 1, 0);
            ctx->traf_p = abcdk_mp4_find2(ctx->moof_p, ABCDK_MP4_ATOM_TYPE_TRAF, 1, 0);
            
traf_next:
            if(!ctx->traf_p)
                goto moof_next;

            ctx->tfhd_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TFHD, 1, 0);
            ctx->tfdt_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TFDT, 1, 0);
            ctx->trun_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TRUN, 1, 0);

            ctx->mfhd = (abcdk_mp4_atom_t *)ctx->mfhd_p->obj->pptrs[0];
            ctx->tfhd = (abcdk_mp4_atom_t *)ctx->tfhd_p->obj->pptrs[0];
            ctx->tfdt = (abcdk_mp4_atom_t *)ctx->tfdt_p->obj->pptrs[0];
            ctx->trun = (abcdk_mp4_atom_t *)ctx->trun_p->obj->pptrs[0];

            if (ctx->tfhd->data.tfhd.trackid != ctx->tkhd->data.tkhd.trackid)
            {
                ctx->traf_p = abcdk_tree_sibling(ctx->traf_p, 0);
                goto traf_next;
            }

            uint32_t offset2 = 0, size = 0;

            offset2 = ctx->trun->data.trun.data_offset;

            lseek(ctx->in_fd, ctx->moof->off_head + offset2, SEEK_SET);

            for (size_t i = 0; i < ctx->trun->data.trun.numbers; i++)
            {
                size = ctx->trun->data.trun.tables[i].sample_size;

                abcdk_mp4_read(ctx->in_fd, ctx->buf, size);

                abcdk_h2645_mp4toannexb(ctx->buf, size, sc_len);
                abcdk_write(ctx->out_fd, ctx->buf, size);
            }

moof_next:
            ctx->moof_p = abcdk_tree_sibling(ctx->moof_p, 0);
        }
    }
    else
    {

        ctx->stsz_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSZ, 1, 1);
        ctx->stss_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSS, 1, 1);
        ctx->stts_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STTS, 1, 1);
        ctx->ctts_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_CTTS, 1, 1);
        ctx->stsc_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSC, 1, 1);
        ctx->stco_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STCO, 1, 1);
                
        ctx->stsz = (abcdk_mp4_atom_t*)(ctx->stsz_p?ctx->stsz_p->obj->pptrs[0]:NULL);
        ctx->stss = (abcdk_mp4_atom_t*)(ctx->stss_p?ctx->stss_p->obj->pptrs[0]:NULL);
        ctx->stts = (abcdk_mp4_atom_t*)(ctx->stts_p?ctx->stts_p->obj->pptrs[0]:NULL);
        ctx->ctts = (abcdk_mp4_atom_t*)(ctx->ctts_p?ctx->ctts_p->obj->pptrs[0]:NULL);
        ctx->stco = (abcdk_mp4_atom_t*)(ctx->stco_p?ctx->stco_p->obj->pptrs[0]:NULL);
        ctx->stsc = (abcdk_mp4_atom_t*)(ctx->stsc_p?ctx->stsc_p->obj->pptrs[0]:NULL);

        for (size_t i = 1; i <= ctx->stsz->data.stsz.numbers; i++)
        {
            uint32_t chunk = 0, offset = 0, id = 0;
            abcdk_mp4_stsc_tell(&ctx->stsc->data.stsc, i, &chunk, &offset, &id);

            uint32_t offset2 = 0, size = 0;
            abcdk_mp4_stsz_tell(&ctx->stsz->data.stsz, offset, i, &offset2, &size);

            lseek(ctx->in_fd, ctx->stco->data.stco.tables[chunk - 1].offset + offset2, SEEK_SET);

            abcdk_mp4_read(ctx->in_fd, ctx->buf, size);
            
            abcdk_h2645_mp4toannexb(ctx->buf,size,sc_len);
            abcdk_write(ctx->out_fd, ctx->buf, size);
        }
    }

final:

    abcdk_h264_extradata_clean(&exdata);
    abcdk_closep(&ctx->out_fd);
}


void _abcdk_m4j_dump_hevc(abcdk_m4j_t *ctx)
{
    uint8_t sc[4];
    int sc_len;
    abcdk_hevc_extradata_t exdata = {0};

    ctx->hvcc_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_HVCC, 1, 1);

    if (!ctx->hvcc_p)
    {
        fprintf(stderr, "HEVC描述信息不存在，忽略当前视频ID(%u)。\n", ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, final);
    }

    ctx->hev1 = (abcdk_mp4_atom_t *)ctx->hev1_p->obj->pptrs[0];
    ctx->hvcc = (abcdk_mp4_atom_t *)ctx->hvcc_p->obj->pptrs[0];

    abcdk_hevc_extradata_deserialize(ctx->hvcc->data.hvcc.extradata->pptrs[0],ctx->hvcc->data.hvcc.extradata->sizes[0],&exdata);

    /*比真实长度少一个字节。*/
    sc_len = exdata.nal_length_size + 1;

    if (sc_len == 3)
        memcpy(sc, "\0\0\1", 3);
    else if (sc_len == 4)
        memcpy(sc, "\0\0\0\1", 4);
    else
    {
        fprintf(stderr, "HEVC仅支持001或0001格式起始码，忽略当前视频ID(%u)。\n", ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, final);
    }

    /*构造文件名。*/
    memset(ctx->out_file, 0, PATH_MAX);
    sprintf(ctx->out_file, "%s/%s-%u.hevc", ctx->save, ctx->in_name, ctx->tkhd->data.tkhd.trackid);

    if (access(ctx->out_file, F_OK) == 0)
    {

        fprintf(stderr, "'%s' 已经存在，忽略当前视频ID(%u)。\n",ctx->out_file,ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, final);
    }

    fprintf(stdout, "%u: %s\n", ctx->tkhd->data.tkhd.trackid, ctx->out_file);

    ctx->out_fd = abcdk_open(ctx->out_file, 1, 0, 1);
    if (ctx->out_fd < 0)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    /*
     * 1：在流的头部写入VPS，SPS，PPS，SEI等。
     * 2：正常的做法是在每个关键帧前都写一次，但会增加流的体积。
    */
    for (int j = 0; j < exdata.nal_array_num; j++)
    {
        struct _nal_array *nal_p = &exdata.nal_array[j];
        for (int k = 0; k < nal_p->nal_num; k++)
        {
            abcdk_write(ctx->out_fd, sc, sc_len);
            abcdk_write(ctx->out_fd, nal_p->nal->pptrs[k], nal_p->nal->sizes[k]);
        }
    }

    if (ctx->mvex_p)
    {
        ctx->moof_p = abcdk_tree_child(ctx->doc, 1);
        while (ctx->moof_p)
        {
            ctx->moof = (abcdk_mp4_atom_t *)ctx->moof_p->obj->pptrs[0];
            if (ctx->moof->type.u32 != ABCDK_MP4_ATOM_TYPE_MOOF)
                goto moof_next;

            ctx->mfhd_p = abcdk_mp4_find2(ctx->moof_p, ABCDK_MP4_ATOM_TYPE_MFHD, 1, 0);
            ctx->traf_p = abcdk_mp4_find2(ctx->moof_p, ABCDK_MP4_ATOM_TYPE_TRAF, 1, 0);

traf_next:
            if(!ctx->traf_p)
                goto moof_next;

            ctx->tfhd_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TFHD, 1, 0);
            ctx->tfdt_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TFDT, 1, 0);
            ctx->trun_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TRUN, 1, 0);

            ctx->mfhd = (abcdk_mp4_atom_t *)ctx->mfhd_p->obj->pptrs[0];
            ctx->tfhd = (abcdk_mp4_atom_t *)ctx->tfhd_p->obj->pptrs[0];
            ctx->tfdt = (abcdk_mp4_atom_t *)ctx->tfdt_p->obj->pptrs[0];
            ctx->trun = (abcdk_mp4_atom_t *)ctx->trun_p->obj->pptrs[0];

            if (ctx->tfhd->data.tfhd.trackid != ctx->tkhd->data.tkhd.trackid)
            {
                ctx->traf_p = abcdk_tree_sibling(ctx->traf_p, 0);
                goto traf_next;
            }

            uint32_t offset2 = 0, size = 0;

            offset2 = ctx->trun->data.trun.data_offset;

            lseek(ctx->in_fd, ctx->moof->off_head + offset2, SEEK_SET);

            for (size_t i = 0; i < ctx->trun->data.trun.numbers; i++)
            {
                size = ctx->trun->data.trun.tables[i].sample_size;

                abcdk_mp4_read(ctx->in_fd, ctx->buf, size);

                abcdk_h2645_mp4toannexb(ctx->buf, size, sc_len);
                abcdk_write(ctx->out_fd, ctx->buf, size);
            }

moof_next:
            ctx->moof_p = abcdk_tree_sibling(ctx->moof_p, 0);
        }
    }
    else
    {

        ctx->stsz_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSZ, 1, 1);
        ctx->stss_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSS, 1, 1);
        ctx->stts_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STTS, 1, 1);
        ctx->ctts_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_CTTS, 1, 1);
        ctx->stsc_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSC, 1, 1);
        ctx->stco_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STCO, 1, 1);
                
        ctx->stsz = (abcdk_mp4_atom_t*)(ctx->stsz_p?ctx->stsz_p->obj->pptrs[0]:NULL);
        ctx->stss = (abcdk_mp4_atom_t*)(ctx->stss_p?ctx->stss_p->obj->pptrs[0]:NULL);
        ctx->stts = (abcdk_mp4_atom_t*)(ctx->stts_p?ctx->stts_p->obj->pptrs[0]:NULL);
        ctx->ctts = (abcdk_mp4_atom_t*)(ctx->ctts_p?ctx->ctts_p->obj->pptrs[0]:NULL);
        ctx->stco = (abcdk_mp4_atom_t*)(ctx->stco_p?ctx->stco_p->obj->pptrs[0]:NULL);
        ctx->stsc = (abcdk_mp4_atom_t*)(ctx->stsc_p?ctx->stsc_p->obj->pptrs[0]:NULL);

        for (size_t i = 1; i <= ctx->stsz->data.stsz.numbers; i++)
        {
            uint32_t chunk = 0, offset = 0, id = 0;
            abcdk_mp4_stsc_tell(&ctx->stsc->data.stsc, i, &chunk, &offset, &id);

            uint32_t offset2 = 0, size = 0;
            abcdk_mp4_stsz_tell(&ctx->stsz->data.stsz, offset, i, &offset2, &size);

            lseek(ctx->in_fd, ctx->stco->data.stco.tables[chunk - 1].offset + offset2, SEEK_SET);

            abcdk_mp4_read(ctx->in_fd, ctx->buf, size);
            
            abcdk_h2645_mp4toannexb(ctx->buf,size,sc_len);
            abcdk_write(ctx->out_fd, ctx->buf, size);
        }
    }

final:

    abcdk_hevc_extradata_clean(&exdata);
    abcdk_closep(&ctx->out_fd);
}

void _abcdk_m4j_dump_acc(abcdk_m4j_t *ctx)
{
    ctx->esds_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_ESDS, 1, 1);

    if (!ctx->esds_p)
    {
        fprintf(stderr, "AAC描述信息不存在，忽略当前音频ID(%u)。\n", ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, final);
    }

    ctx->mp4a = (abcdk_mp4_atom_t *)(ctx->mp4a_p->obj->pptrs[0]);
    ctx->esds = (abcdk_mp4_atom_t *)(ctx->esds_p->obj->pptrs[0]);

    memset(ctx->out_file,0,PATH_MAX);
    sprintf(ctx->out_file,"%s/%s-%u.aac",ctx->save,ctx->in_name,ctx->tkhd->data.tkhd.trackid);

    if (access(ctx->out_file, F_OK) == 0)
    {
        fprintf(stderr, "'%s' 已经存在，忽略当前音频ID(%u)。\n",ctx->out_file,ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, final);
    }

    fprintf(stdout, "%u: %s\n", ctx->tkhd->data.tkhd.trackid, ctx->out_file);

    ctx->out_fd = abcdk_open(ctx->out_file, 1, 0, 1);
    if (ctx->out_fd < 0)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    /*解析ADTS信息。*/
#if 0
    _abcdk_m4j_aac_decode_extradata(ctx,ctx->esds->data.esds.dec_sp_info.extradata->pptrs[0],
                                   ctx->esds->data.esds.dec_sp_info.extradata->sizes[0]);

#else
    abcdk_aac_extradata_deserialize(ctx->esds->data.esds.dec_sp_info.extradata->pptrs[0],
                                     ctx->esds->data.esds.dec_sp_info.extradata->sizes[0],
                                     &ctx->adts_hdr);
#endif

    if(ctx->mvex_p)
    {
        ctx->moof_p = abcdk_tree_child(ctx->doc, 1);
        while (ctx->moof_p)
        {
            ctx->moof = (abcdk_mp4_atom_t *)ctx->moof_p->obj->pptrs[0];
            if (ctx->moof->type.u32 != ABCDK_MP4_ATOM_TYPE_MOOF)
                goto moof_next;

            ctx->mfhd_p = abcdk_mp4_find2(ctx->moof_p, ABCDK_MP4_ATOM_TYPE_MFHD, 1, 0);
            ctx->traf_p = abcdk_mp4_find2(ctx->moof_p, ABCDK_MP4_ATOM_TYPE_TRAF, 1, 0);

traf_next:
            if(!ctx->traf_p)
                goto moof_next;

            ctx->tfhd_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TFHD, 1, 0);
            ctx->tfdt_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TFDT, 1, 0);
            ctx->trun_p = abcdk_mp4_find2(ctx->traf_p, ABCDK_MP4_ATOM_TYPE_TRUN, 1, 0);

            ctx->mfhd = (abcdk_mp4_atom_t *)ctx->mfhd_p->obj->pptrs[0];
            ctx->tfhd = (abcdk_mp4_atom_t *)ctx->tfhd_p->obj->pptrs[0];
            ctx->tfdt = (abcdk_mp4_atom_t *)ctx->tfdt_p->obj->pptrs[0];
            ctx->trun = (abcdk_mp4_atom_t *)ctx->trun_p->obj->pptrs[0];

            if (ctx->tfhd->data.tfhd.trackid != ctx->tkhd->data.tkhd.trackid)
            {
                ctx->traf_p = abcdk_tree_sibling(ctx->traf_p, 0);
                goto traf_next;
            }

            uint32_t offset2 = 0, size = 0;

            offset2 = ctx->trun->data.trun.data_offset;

            lseek(ctx->in_fd, ctx->moof->off_head + offset2, SEEK_SET);

            for (size_t i = 0; i < ctx->trun->data.trun.numbers; i++)
            {
                size = ctx->trun->data.trun.tables[i].sample_size;

                abcdk_mp4_read(ctx->in_fd, ctx->buf, size);

                /*每帧都要加7字节的帧头。*/
                char hdr[7] = {0};

                ctx->adts_hdr.aac_frame_length = 7 + size; // size是数据帧的大小。
                abcdk_aac_adts_header_serialize(&ctx->adts_hdr, hdr, 7);

                abcdk_write(ctx->out_fd, hdr, 7);
                abcdk_write(ctx->out_fd, ctx->buf, size);
            }

moof_next:
            ctx->moof_p = abcdk_tree_sibling(ctx->moof_p, 0);
        }
    }
    else
    {

        ctx->stsz_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSZ, 1, 1);
        ctx->stss_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSS, 1, 1);
        ctx->stts_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STTS, 1, 1);
        ctx->ctts_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_CTTS, 1, 1);
        ctx->stsc_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STSC, 1, 1);
        ctx->stco_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_STCO, 1, 1);
                
        ctx->stsz = (abcdk_mp4_atom_t*)(ctx->stsz_p?ctx->stsz_p->obj->pptrs[0]:NULL);
        ctx->stss = (abcdk_mp4_atom_t*)(ctx->stss_p?ctx->stss_p->obj->pptrs[0]:NULL);
        ctx->stts = (abcdk_mp4_atom_t*)(ctx->stts_p?ctx->stts_p->obj->pptrs[0]:NULL);
        ctx->ctts = (abcdk_mp4_atom_t*)(ctx->ctts_p?ctx->ctts_p->obj->pptrs[0]:NULL);
        ctx->stco = (abcdk_mp4_atom_t*)(ctx->stco_p?ctx->stco_p->obj->pptrs[0]:NULL);
        ctx->stsc = (abcdk_mp4_atom_t*)(ctx->stsc_p?ctx->stsc_p->obj->pptrs[0]:NULL);

        for (size_t i = 1; i <= ctx->stsz->data.stsz.numbers; i++)
        {
            uint32_t chunk = 0, offset = 0, id = 0;
            abcdk_mp4_stsc_tell(&ctx->stsc->data.stsc, i, &chunk, &offset, &id);

            uint32_t offset2 = 0, size = 0;
            abcdk_mp4_stsz_tell(&ctx->stsz->data.stsz, offset, i, &offset2, &size);

            lseek(ctx->in_fd, ctx->stco->data.stco.tables[chunk - 1].offset + offset2, SEEK_SET);

            abcdk_mp4_read(ctx->in_fd, ctx->buf, size);

            char hdr[7] = {0};
                    
            ctx->adts_hdr.aac_frame_length = 7+size;//size是数据帧的大小。
            abcdk_aac_adts_header_serialize(&ctx->adts_hdr,hdr,7);

            abcdk_write(ctx->out_fd, hdr, 7);
            abcdk_write(ctx->out_fd, ctx->buf, size);
        }
    }

final:

    abcdk_closep(&ctx->out_fd);


}

void _abcdk_m4j_dump_video(abcdk_m4j_t *ctx)
{
    ctx->avc1_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_AVC1, 1, 1);
    ctx->hev1_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_HEV1, 1, 1);

    if (!ctx->avc1_p && !ctx->hev1_p)
    {
        fprintf(stderr, "仅支持H264或HEVC编码提取，忽略当前视频ID(%u)。\n", ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_RETURN0(ctx->errcode = 0);
    }

    if(ctx->avc1_p)
        _abcdk_m4j_dump_h264(ctx);
    if(ctx->hev1_p)
        _abcdk_m4j_dump_hevc(ctx);
}

void _abcdk_m4j_dump_audio(abcdk_m4j_t *ctx)
{
    ctx->mp4a_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_MP4A, 1, 1);

    if (!ctx->mp4a_p)
    {
        fprintf(stderr, "仅支持AAC编码提取，忽略当前音频ID(%u)。\n", ctx->tkhd->data.tkhd.trackid);
        ABCDK_ERRNO_AND_RETURN0(ctx->errcode = 0);
    }

    if(ctx->mp4a_p)
        _abcdk_m4j_dump_acc(ctx);
}

void _abcdk_m4j_dump(abcdk_m4j_t *ctx)
{
    ctx->out_fd = -1;

    ctx->moov_p = abcdk_mp4_find2(ctx->doc,ABCDK_MP4_ATOM_TYPE_MOOV,1,1);
    if(!ctx->moov_p)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = ESPIPE, final);

    /*查找mvex，用于区别MP4和FMP4。*/
    ctx->mvex_p = abcdk_mp4_find2(ctx->moov_p,ABCDK_MP4_ATOM_TYPE_MVEX,1,1);

    for (int i = 0; i < 1000; i++)
    {
        ctx->trak_p = abcdk_mp4_find2(ctx->moov_p, ABCDK_MP4_ATOM_TYPE_TRAK, i + 1, 0);
        if (!ctx->trak_p)
            ABCDK_ERRNO_AND_GOTO1(0, final);

        ctx->tkhd_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_TKHD, 1, 1);
        ctx->hdlr_p = abcdk_mp4_find2(ctx->trak_p, ABCDK_MP4_ATOM_TYPE_HDLR, 1, 1);

        if(!ctx->tkhd_p || !ctx->hdlr_p)
            ABCDK_ERRNO_AND_GOTO1(1, final);

        ctx->tkhd = (abcdk_mp4_atom_t*)ctx->tkhd_p->obj->pptrs[0];
        ctx->hdlr = (abcdk_mp4_atom_t*)ctx->hdlr_p->obj->pptrs[0];

        if (ctx->hdlr->data.hdlr.subtype.u32 == ABCDK_MP4_ATOM_MKTAG('v', 'i', 'd', 'e') &&
            !ctx->ignore_video)
        {
            _abcdk_m4j_dump_video(ctx);
        }
        else if (ctx->hdlr->data.hdlr.subtype.u32 == ABCDK_MP4_ATOM_MKTAG('s', 'o', 'u', 'n') &&
                 !ctx->ignore_audio)
        {
            _abcdk_m4j_dump_audio(ctx);
        }

        /*有错误发生，提前终止。*/
        if(ctx->errcode)
            break;
    }

final:

    abcdk_closep(&ctx->out_fd);

}

void _abcdk_m4j_work(abcdk_m4j_t *ctx)
{
    ctx->in_fd = -1;

    ctx->file = abcdk_option_get(ctx->args, "--file", 0, NULL);
    ctx->save = abcdk_option_get(ctx->args, "--save", 0, "./");
    ctx->ignore_video = abcdk_option_exist(ctx->args, "--ignore-video");
    ctx->ignore_audio = abcdk_option_exist(ctx->args, "--ignore-audio");

    if (!ctx->file || !*ctx->file)
    {
        fprintf(stderr, "'--file FILE' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->file, R_OK) != 0)
    {
        fprintf(stderr, "'%s' %s.\n", ctx->file, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    if (!ctx->save || !*ctx->save)
    {
        fprintf(stderr, "'--save PATH' 不能省略，且不能为空。\n");
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EINVAL, final);
    }

    if (access(ctx->save, W_OK) != 0)
    {
        fprintf(stderr, "'%s' %s.\n", ctx->save, strerror(errno));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);
    }

    ctx->out_file = abcdk_heap_alloc(PATH_MAX);
    if(!ctx->out_file)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    ctx->buf_size = 16 * 1024 * 1024; //希望够用。
    ctx->buf = abcdk_heap_alloc(ctx->buf_size);
    if(!ctx->buf)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    ctx->in_fd = abcdk_open(ctx->file, 0, 0, 0);
    if (ctx->in_fd < 0)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    ctx->doc = abcdk_mp4_read_probe2(ctx->in_fd, 0, -1UL, 0);
    if (!ctx->doc)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = errno, final);

    if(!abcdk_mp4_find2(ctx->doc,ABCDK_MP4_ATOM_TYPE_FTYP,1,1))
    {
        fprintf(stderr, "'%s' 可能不是MP4文件。\n", ctx->file);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = EPERM, final);
    }

    memset(ctx->in_name,0,sizeof(ctx->in_name));
    abcdk_basename(ctx->in_name,ctx->file);

    _abcdk_m4j_dump(ctx);

final:

    abcdk_heap_free(ctx->buf);
    abcdk_heap_free(ctx->out_file);
    abcdk_closep(&ctx->in_fd);
    abcdk_tree_free(&ctx->doc);
}

int abcdk_tool_mp4juicer(abcdk_option_t *args)
{
    abcdk_m4j_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_m4j_print_usage(ctx.args, 0);
    }
    else
    {
        _abcdk_m4j_work(&ctx);
    }

    return ctx.errcode;
}