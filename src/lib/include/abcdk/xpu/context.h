/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_CONTEXT_H
#define ABCDK_XPU_CONTEXT_H

#include "abcdk/xpu/types.h"


__BEGIN_DECLS

/**设备环境.*/
typedef struct _abcdk_xpu_context abcdk_xpu_context_t;

/**
 * 绑定到当前线程.
 * 
 * @note 仅对当前线程有效, 其它线程不可见. 允许为NULL(0).
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_xpu_context_current_set(abcdk_xpu_context_t *ctx);


/**释放.*/
void abcdk_xpu_context_unref(abcdk_xpu_context_t **ctx);

/**引用.*/
abcdk_xpu_context_t *abcdk_xpu_context_refer(abcdk_xpu_context_t *ctx);

/**创建.*/
abcdk_xpu_context_t *abcdk_xpu_context_alloc(int id);




__END_DECLS

#endif // ABCDK_XPU_CONTEXT_H