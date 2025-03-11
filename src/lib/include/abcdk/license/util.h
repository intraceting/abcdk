/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LICENSE_UTIL_H
#define ABCDK_LICENSE_UTIL_H

#include "abcdk/license/license.h"

__BEGIN_DECLS

/**
 * 打印授权信息。
 */
void abcdk_license_dump(const abcdk_license_info_t *info);

/**
 * 状态。
 *
 * @param [in] realtime 自然时间(秒，UTC)。
 *
 * @return > 0 剩余时长(秒)，-1 已过期或未生效。
 */
int64_t abcdk_license_status(const abcdk_license_info_t *info, uint64_t realtime, int dump_if_expire);


__END_DECLS

#endif //ABCDK_LICENSE_UTIL_H