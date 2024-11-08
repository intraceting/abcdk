/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_CONTEXT_H
#define ABCDK_UTIL_CONTEXT_H

#include "abcdk/util/general.h"
#include "abcdk/util/heap.h"
#include "abcdk/util/object.h"
#include "abcdk/util/io.h"
#include "abcdk/util/rwlock.h"

__BEGIN_DECLS

/**简单的上下文环境。 */
typedef struct _abcdk_context abcdk_context_t;

/**释放。*/
void abcdk_context_unref(abcdk_context_t **ctx);

/**引用。*/
abcdk_context_t *abcdk_context_refer(abcdk_context_t *src);

/**申请。*/
abcdk_context_t *abcdk_context_alloc(size_t userdata, void (*free_cb)(void *userdata));

/** 获取用户环境指针。*/
void *abcdk_context_get_userdata(abcdk_context_t *ctx);

/**读锁。 */
void abcdk_context_rdlock(abcdk_context_t *ctx);

/**写锁。 */
void abcdk_context_wrlock(abcdk_context_t *ctx);

/**解锁。 */
int abcdk_context_unlock(abcdk_context_t *ctx,int exitcode);

__END_DECLS

#endif //ABCDK_UTIL_CONTEXT_H
