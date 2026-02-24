/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_RUNTIME_H
#define ABCDK_XPU_RUNTIME_H

#include "abcdk/xpu/types.h"

__BEGIN_DECLS


/**去初始化.*/
int abcdk_xpu_runtime_deinit();

/**
 * 初始化.
 *
 * @return 0 成功, < 0  失败.
 */
int abcdk_xpu_runtime_init(int hwaccel, ...);

__END_DECLS

#endif // ABCDK_XPU_RUNTIME_H
