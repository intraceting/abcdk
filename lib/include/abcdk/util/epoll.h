/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_UTIL_EPOLL_H
#define ABCDK_UTIL_EPOLL_H

#include "abcdk/util/general.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/**
 * EOPLL的事件。
*/
typedef enum _abcdk_epoll_events
{
    /**
     * 输入事件。
    */
    ABCDK_EPOLL_INPUT = 0x00000001,
#define ABCDK_EPOLL_INPUT ABCDK_EPOLL_INPUT

    /**
     * 输出事件。
    */
    ABCDK_EPOLL_OUTPUT = 0x00000100,
#define ABCDK_EPOLL_OUTPUT ABCDK_EPOLL_OUTPUT

    /**
     * 出错事件。
    */
    ABCDK_EPOLL_ERROR = 0x01000000
#define ABCDK_EPOLL_ERROR ABCDK_EPOLL_ERROR
}abcdk_epoll_events_t;

/**
 * 事件结构体。
 * 
 * @note 不能使用原始事件的值。
*/
typedef struct epoll_event abcdk_epoll_event_t;

/**
 * 创建EPOLL句柄
 * 
 * @return >=0 成功(EPOLL句柄)，-1 失败。
*/
int abcdk_epoll_create();

/**
 * 注册句柄(文件或SOCKET)和事件。
 * 
 * @param [in] first 是否首次注册。0 否，!0 是。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_epoll_mark(int efd, int fd,const abcdk_epoll_event_t *event, int first);

/**
 * 删除句柄(文件或SOCKET)和事件。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_epoll_drop(int efd, int fd);

/**
 * 等待事件。
 * 
 * @param [in] timeout 超时(毫秒)。>= 0 有事件或时间过期，< 0 直到有事件或出错。
 * 
 * @return > 0 事件数量，<= 0 超时或出错。
*/
int abcdk_epoll_wait(int efd,abcdk_epoll_event_t *events,int max,time_t timeout);

__END_DECLS

#endif //ABCDK_UTIL_EPOLL_H