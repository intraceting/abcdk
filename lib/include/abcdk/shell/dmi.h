/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SHELL_DMI_H
#define ABCDK_SHELL_DMI_H

#include "abcdk/util/block.h"

/**
 * 获取硬件散列值。
 * 
 * @param [in] salt 盐(干扰项，隆低不同机器之间碰撞机率。)。NULL(0) 忽略。
*/
const char *abcdk_dmi_get_machine_hashcode(char buf[37],const char *salt, ...);

#endif //ABCDK_SHELL_DMI_H