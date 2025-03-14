/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LICENSE_LICENSE_H
#define ABCDK_LICENSE_LICENSE_H

#include "abcdk/util/bit.h"
#include "abcdk/util/basecode.h"
#include "abcdk/util/object.h"
#include "abcdk/util/clock.h"
#include "abcdk/util/trace.h"
#include "abcdk/openssl/cipher.h"

__BEGIN_DECLS

/**简单的许可证信息。*/
typedef struct _abcdk_license_info
{
    /**类别。*/
    uint8_t category;

    /**型号。*/
    uint8_t product;

    /**有效期限(天)。*/
    uint16_t duration;

    /**启用时间(秒，UTC)。*/
    uint64_t begin;

} abcdk_license_info_t;


#endif // ABCDK_LICENSE_LICENSE_H
