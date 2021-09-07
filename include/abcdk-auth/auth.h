/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_AUTH_AUTH_H
#define ABCDK_AUTH_AUTH_H

#include "abcdk-util/general.h"
#include "abcdk-util/getargs.h"
#include "abcdk-util/socket.h"

__BEGIN_DECLS

/**
 * 收集DMI信息。
 * 
 * @note 仅包括system-serial-number和system-uuid。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_collect_dmi(abcdk_tree_t *opt);

/**
 * 收集MAC地址。
 * 
 * @note 仅包括物理地址。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_collect_mac(abcdk_tree_t *opt);

/**
 * 生成有效期限。
 * 
 * @param days 使用天数。
 * @param begin 开始日期(UTC)，NULL(0) 当前时间。
*/
int abcdk_auth_make_valid_period(abcdk_tree_t *opt, uintmax_t days, struct tm *begin);

__END_DECLS

#endif //ABCDK_AUTH_AUTH_H
