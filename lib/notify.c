/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/notify.h"

int abcdk_notify_init(int nonblock)
{
    int flags = IN_CLOEXEC;

    if(nonblock)
        flags |= IN_NONBLOCK;

    return inotify_init1(flags);
}

int abcdk_notify_add(int fd, const char *name, uint32_t masks)
{
    assert(fd >= 0 && name != NULL && (masks & IN_ALL_EVENTS));

    return inotify_add_watch(fd, name, masks & IN_ALL_EVENTS);
}

int abcdk_notify_remove(int fd,int wd)
{
    assert(fd >= 0);

    return inotify_rm_watch(fd,wd);
}

int abcdk_notify_watch(int fd,abcdk_notify_event_t *event,time_t timeout)
{
    ssize_t rlen = 0;

    assert(fd >= 0 && event != NULL);
    assert(event->buf != NULL && event->buf->data != NULL && event->buf->size > 0);

    /* 清除已经读取的。*/
    abcdk_buffer_drain(event->buf);

    /*缓存无数据，先等待。*/
    if (event->buf->wsize <= 0)
    {
        /* 等待事件到来。 */
        if (abcdk_poll(fd, 0x01, timeout) <= 0)
            return -1;

        /* 导入缓存。 */
        if( abcdk_buffer_import_atmost(event->buf,fd,SIZE_MAX)<=0)
            return -1;
    }

    /*读事件，必须符合事件结构大小，不能多，也不能少。*/
    rlen = abcdk_buffer_read(event->buf, &event->event, sizeof(struct inotify_event));
    if (rlen <= 0 || rlen != sizeof(struct inotify_event))
        return -1;

    /* 事件后没有附带名字，直接返回。*/
    if(event->event.len<=0)
        return 0;

    /* 事件后跟着固定长度的名字，读取的名字长度符合事件描述的大小，不能多，也不能少。*/
    rlen = abcdk_buffer_read(event->buf,event->name,event->event.len);
    if (rlen <= 0 || rlen != event->event.len)
        return -1;

    return 0;
}