/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_LOGON_H
#define ABCDK_UTIL_LOGON_H

#include "abcdk/util/general.h"
#include "abcdk/util/map.h"
#include "abcdk/util/rwlock.h"
#include "abcdk/util/context.h"

__BEGIN_DECLS

/**简单的登录容器。 */
typedef struct _abcdk_logon abcdk_logon_t;

/**配置。*/
typedef struct _abcdk_logon_config
{
    /**启用监视。!0 是，0 否。*/
    int enable_watch;

    /**环境指针。*/
    void *opaque;

    /**
     * KEY长度回调函数。
     * 
     * @note 默认按字符串处理。
     */
    uint64_t (*key_size_cb)(const void *key, void *opaque);

    /**
     * KEY哈希回调函数。
     * 
     * @note 默认使用BKDRHash64算法。
     */
    uint64_t (*key_hash_cb)(const void *key, void *opaque);

    /**
     * KEY比较回调函数。
     * 
     * @note 默认按ASCII值比较。
     *
     * @return 0(key1 == key2)，!0(key1 != key2)。
     */
    int (*key_compare_cb)(const void *key1,  const void *key2, void *opaque);

    /**KEY删除回调函数。*/
    void (*key_remove_cb)(const void *key, abcdk_context_t *userdata, void *opaque);

}abcdk_logon_config_t;

/**销毁。 */
void abcdk_logon_destroy(abcdk_logon_t **ctx);

/**创建。 */
abcdk_logon_t *abcdk_logon_create(abcdk_logon_config_t *cfg);

/**删除。*/
void abcdk_logon_remove(abcdk_logon_t *ctx,const void *key);

/**
 * 添加。
 * 
 * @note 如果不存在，则自动创建。
 * @note 返回的用户环境指针仅为指针复制，没有增加引用计数。
 * 
 * @param [in] userdata 用户环境大小。= 0 仅查询。
 * 
 * @return !NULL(0) 成功(用户环境指针)，NULL(0) 失败。
 * 
*/
abcdk_context_t *abcdk_logon_insert(abcdk_logon_t *ctx,const void *key,size_t userdata);

/**
 * 查询。
 * 
 * @note 返回的用户环境指针仅为指针复制，没有增加引用计数。
 * 
 * @return !NULL(0) 成功(用户环境指针)，NULL(0) 失败。
*/
abcdk_context_t *abcdk_logon_lookup(abcdk_logon_t *ctx,const void *key);

/**
 * 监视(遍历)。
 * 
 * @note 返回的用户环境指针仅为指针复制，没有增加引用计数。
 * 
 * @param [in out] it 迭代器。NULL(0) 表示从头部开始遍历。
 * 
 * @return !NULL(0) 数据指针。NULL(0) 结束。
 */
abcdk_context_t *abcdk_logon_next(abcdk_logon_t *ctx,void **it);

/**读锁。 */
void abcdk_logon_rdlock(abcdk_logon_t *ctx);

/**写锁。 */
void abcdk_logon_wrlock(abcdk_logon_t *ctx);

/**解锁。 */
int abcdk_logon_unlock(abcdk_logon_t *ctx,int exitcode);


__END_DECLS

#endif //ABCDK_UTIL_LOGON_H