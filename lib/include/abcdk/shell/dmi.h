/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_DMI_H
#define ABCDK_SHELL_DMI_H

#include "abcdk/shell/block.h"
#include "abcdk/util/socket.h"
#include "abcdk/util/md5.h"

/**
 * 获取硬件散列值。
 * 
 * @param [in] flag 标志，忽略。
*/
const char *abcdk_dmi_get_machine_hashcode(uint8_t uuid[16], int flag,...);

#endif //ABCDK_SHELL_DMI_H