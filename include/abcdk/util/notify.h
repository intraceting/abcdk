/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_NOTIFY_H
#define ABCDK_UTIL_NOTIFY_H

#include "abcdk/util/general.h"
#include "abcdk/util/buffer.h"

__BEGIN_DECLS

/**
 * 简单的事件变更监视器。
*/
typedef struct _abcdk_notify_event
{
    /** 
     * 缓存。
     * 
     * @note 尽量不要直接修改。
    */
    abcdk_buffer_t *buf;

    /** 监视ID。*/
    int wd;

    /** 事件掩码。*/
    uint32_t mask;	

    /** 如果两个监视ID之间存在关联，则这个值是相同的。*/
    uint32_t cookie;

    /** 名字。*/
    const char *name;

} abcdk_notify_event_t;

/**
 * 释放事件对象。
*/
void abcdk_notify_free(abcdk_notify_event_t **event);

/**
 * 申请事件对象。
*/
abcdk_notify_event_t *abcdk_notify_alloc(size_t buf_size);

/**
 * 初始化监视器。
 * 
 * @param nonblock 0 阻塞模式，!0 非阻塞模式。
 * 
 * @return >= 0 句柄，< 0 错误。
*/
int abcdk_notify_init(int nonblock);

/**
 * 添加一个监视对象(文件或目录)。
 * 
 * @return >= 0 成功(监视ID)，< 0 错误。
*/
int abcdk_notify_add(int fd,const char* name,uint32_t masks);

/**
 * 删除一个监视对象。
 * 
 * @param [in] wd 监视ID。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_notify_remove(int fd,int wd);

/**
 * 监视变更事件。
 * 
 * @note 阻塞模式的句柄，响应可能会有延迟。
 * 
 * @param [in] fd 句柄。
 * @param [out] event 事件。
 * @param [in] timeout 超时(毫秒)。
 * 
 * @return 0 成功，!0 超时或失败。
*/
int abcdk_notify_watch(int fd,abcdk_notify_event_t *event,time_t timeout);


__END_DECLS

#endif //ABCDK_UTIL_NOTIFY_H