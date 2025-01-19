/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LICENCE_LICENCE_H
#define ABCDK_LICENCE_LICENCE_H

#include "abcdk/util/bit.h"
#include "abcdk/openssl/cipher.h"

__BEGIN_DECLS

#ifdef OPENSSL_VERSION_NUMBER

/**简单的许可证信息。*/
typedef struct _abcdk_license
{
    /**类别。*/
    uint8_t category;

    /**型号。*/
    uint8_t product;

    /**节点数量(个)。*/
    uint16_t node;
    
    /**有效期限(天)。*/
    uint16_t duration;

    /**启用时间(秒，UTC)。*/
    uint64_t begin;

}abcdk_license_t;


/**
 * 生成授权码。
 * 
 * @param [in] passwd 密钥。
*/
abcdk_object_t *abcdk_license_generate(const abcdk_license_t *info,const char *passwd);

/**
 * 解包。
 * 
 * @return 0 成功，-1 失败(无效的或错误的)。
*/
int abcdk_license_unpack(abcdk_license_t *info,const char *sn,const char *passwd);

/**
 * 打印授权信息。
 */
void abcdk_license_dump(const abcdk_license_t *info);

/**
 * 状态。
 * 
 * @param [in] realtime 自然时间(秒，UTC)。
 * 
 * @return > 0 剩余时长(秒)，-1 已过期或未生效。
*/
int64_t abcdk_license_status(uint64_t realtime, const abcdk_license_t *info, int dump_if_expire);

#endif //OPENSSL_VERSION_NUMBER

__END_DECLS

#endif //ABCDK_LICENCE_LICENCE_H
