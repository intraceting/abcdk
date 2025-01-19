/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/license/license.h"

#ifdef OPENSSL_VERSION_NUMBER

/*
 * ------------------------------------------------------------------|
 * |Licence Data                                                     |
 * ------------------------------------------------------------------|
 * |Version |Category |Product |Begin   |Durations |Nodes   |Reserve |
 * |1 byte  |1 byte   |1 byte  |8 Bytes |2 Bytes   |2 Bytes |2 Bytes |
 * ------------------------------------------------------------------|
*/

#define ABCDK_LICENSE_VER_1_SRC_LEN (1 + 1 + 1 + 8 + 2 + 2 + 2)

static abcdk_object_t *_abcdk_license_update(void *src, size_t slen, const char *passwd,int enc)
{
    abcdk_openssl_cipher_t *ctx;
    abcdk_object_t *dst = NULL;
    int chk;
    
    ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM,(uint8_t*)passwd,strlen(passwd));
    if(!ctx)
        return NULL;

    dst = abcdk_openssl_cipher_update_pack(ctx,(uint8_t*)src,slen,enc);
    if(!dst)
        goto ERR;

    abcdk_openssl_cipher_destroy(&ctx);
    return dst;

ERR:

    abcdk_openssl_cipher_destroy(&ctx);
    abcdk_object_unref(&dst);
    return NULL;
}


abcdk_object_t *abcdk_license_generate(const abcdk_license_t *info,const char *passwd)
{
    uint8_t src[ABCDK_LICENSE_VER_1_SRC_LEN] = {0};
    abcdk_bit_t src_bit = {0};
    uint32_t src_chksum = ~0;
    abcdk_object_t *dst = NULL;
    abcdk_object_t *dst2 = NULL;

    assert(info != NULL && passwd != NULL);

    src_bit.data = src;
    src_bit.size = ABCDK_LICENSE_VER_1_SRC_LEN;

    abcdk_bit_write_number(&src_bit, 8, 1);
    abcdk_bit_write_number(&src_bit, 8, info->category);
    abcdk_bit_write_number(&src_bit, 8, info->product);
    abcdk_bit_write_number(&src_bit, 64, info->begin);
    abcdk_bit_write_number(&src_bit, 16, info->duration);
    abcdk_bit_write_number(&src_bit, 16, info->node);
    abcdk_bit_seek(&src_bit,16);

    dst = _abcdk_license_update(src_bit.data, src_bit.pos / 8, passwd,1);
    if (!dst)
        return NULL;

    dst2 = abcdk_basecode_encode2(dst->pptrs[0], dst->sizes[0], 64);
    abcdk_object_unref(&dst);

    return dst2;
}

int abcdk_license_unpack(abcdk_license_t *info,const char *sn,const char *passwd)
{
    abcdk_object_t *src = NULL;
    abcdk_object_t *src2 = NULL;
    abcdk_bit_t src_bit = {0};
    uint8_t src_salt[256] = {0};
    uint32_t old_chksum = 0;
    uint32_t new_chksum = 0;
    uint8_t ver;
    
    assert(info != NULL && sn != NULL && passwd != NULL);

    src = abcdk_basecode_decode2(sn,strlen(sn),64);
    if(!src)
        goto ERR;

    src2 = _abcdk_license_update(src->pptrs[0],src->sizes[0],passwd,0);
    if(!src2)
        goto ERR;

    if(ABCDK_LICENSE_VER_1_SRC_LEN != src2->sizes[0])
        goto ERR;

    src_bit.data = src2->pptrs[0];
    src_bit.size = src2->sizes[0];

    ver = abcdk_bit_read2number(&src_bit,8);
    if(ver != 1)
        goto ERR;
    
    info->category = abcdk_bit_read2number(&src_bit,8);
    info->product = abcdk_bit_read2number(&src_bit,8);
    info->begin = abcdk_bit_read2number(&src_bit,64);
    info->duration = abcdk_bit_read2number(&src_bit,16);
    info->node = abcdk_bit_read2number(&src_bit,16);

    abcdk_object_unref(&src);
    abcdk_object_unref(&src2);
    return 0;

ERR:

    abcdk_object_unref(&src);
    abcdk_object_unref(&src2);
    return -1;
}

void abcdk_license_dump(const abcdk_license_t *info)
{
    char msg[1024] = {0};
    struct tm begin_tm = {0}, end_tm = {0};

    assert(info != NULL);

    /*转换起止时间。*/
    abcdk_time_sec2tm(&begin_tm, info->begin, 0);
    abcdk_time_sec2tm(&end_tm, info->begin + info->duration * 24 * 3600ULL, 0);

    sprintf(msg + strlen(msg), "产品类别(%hhd)；", info->category);
    sprintf(msg + strlen(msg), "产品型号(%hhd)；", info->product);
    sprintf(msg + strlen(msg), "节点数量(%hu)；", info->node);
    sprintf(msg + strlen(msg), "生效日期(%04d年%02d月%02d日)；", begin_tm.tm_year + 1900, begin_tm.tm_mon + 1, begin_tm.tm_mday);
    sprintf(msg + strlen(msg), "终止日期(%04d年%02d月%02d日)；", end_tm.tm_year + 1900, end_tm.tm_mon + 1, end_tm.tm_mday);

    abcdk_trace_printf(LOG_INFO, "授权摘要：%s\n", msg);

    return ;
}

static int64_t _abcdk_license_duration(uint64_t realtime, const abcdk_license_t *info)
{
    ssize_t dlen = 0;
    char dst[100] = {0};
    uint64_t valid_secs = 0;
    int64_t remainder = 0;
    int chk;

    assert(info != NULL);

    /*有效期限由天转换成秒。*/
    valid_secs = info->duration  * 24 * 60 * 60LU;

    /*开始时间大于当前时间，则未生效。*/
    if (info->begin > realtime)
        return -1;

    /*结束时间小于当前时间，则已过期。*/
    if (info->begin + valid_secs < realtime)
        return -1;

    /*运行时长不能超过有效时长。*/
    if (valid_secs <= (realtime - info->begin))
        return -1;

    /*计算剩余秒数。*/
    remainder = valid_secs - (realtime - info->begin);

    return remainder;
}

int64_t abcdk_license_status(uint64_t realtime, const abcdk_license_t *info,int dump_if_expire)
{
    int64_t remain_sec = 0;

    assert(info != NULL);

    remain_sec = _abcdk_license_duration(realtime,info);

    if(remain_sec <= 0 && dump_if_expire)
        abcdk_license_dump(info);

    return remain_sec;
}


#endif //OPENSSL_VERSION_NUMBER