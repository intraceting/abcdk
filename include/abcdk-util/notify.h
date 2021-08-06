/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_NOTIFY_H
#define ABCDK_UTIL_NOTIFY_H

#include "abcdk-util/general.h"
#include "abcdk-util/buffer.h"

__BEGIN_DECLS

/**
 * 监视器事件
 * 
*/
typedef struct _abcdk_notify_event
{
    /**
     * 缓存。
     * 
     * 由调用者申请和释放。
    */
    abcdk_buffer_t *buf;

    /**
     * 事件。
    */
    struct inotify_event event;

    /** 
     * 名字。
    */
    char name[PATH_MAX];

} abcdk_notify_event_t;

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
 * @return >= 0 成功(WD)，< 0 错误。
*/
int abcdk_notify_add(int fd,const char* name,uint32_t masks);

/**
 * 删除一个监视对象。
 * 
 * @return 0 成功，!0 失败。
*/
int abcdk_notify_remove(int fd,int wd);

/**
 * 监视。
 * 
 * 阻塞模式的句柄，响应可能会有延迟。
 * 
 * @param timeout 超时(毫秒)。>= 0 有事件或时间过期，< 0 直到事件或出错。
 * 
 * @return 0 成功，!0 超时或失败。
*/
int abcdk_notify_watch(int fd,abcdk_notify_event_t *event,time_t timeout);


__END_DECLS

#endif //ABCDK_UTIL_NOTIFY_H