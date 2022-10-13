/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/util/notify.h"

void abcdk_notify_free(abcdk_notify_event_t **event)
{
    abcdk_notify_event_t *event_p = NULL;

    if(!event || !*event)
        return;

    event_p = *event;
    *event = NULL;

    abcdk_buffer_free(&event_p->buf);
    abcdk_heap_free2((void**)&event_p->name);
    abcdk_heap_free(event_p);
}

abcdk_notify_event_t *abcdk_notify_alloc(size_t buf_size)
{
    abcdk_notify_event_t *event = NULL;

    if (buf_size < 4096)
        buf_size = 4096;

    event = abcdk_heap_alloc(sizeof(abcdk_notify_event_t));
    if(!event)
        return NULL;

    event->buf = abcdk_buffer_alloc2(buf_size);
    if(!event->buf)
        goto final_error;

    event->name = abcdk_heap_alloc(PATH_MAX);
    if(!event->name)
        goto final_error;

    return event;

final_error:

    abcdk_notify_free(&event);

    return NULL;
}

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
    struct inotify_event et = {0};

    assert(fd >= 0 && event != NULL);

    /*清除已经读取的。*/
    abcdk_buffer_drain(event->buf);

    memset((char*)event->name,0,PATH_MAX);

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
    rlen = abcdk_buffer_read(event->buf, &et, sizeof(et));
    if (rlen <= 0 || rlen != sizeof(et))
        return -1;

    /* 事件后没有附带名字，直接返回。*/
    if(et.len<=0)
        return 0;

    /* 事件后跟着固定长度的名字，读取的名字长度符合事件描述的大小，不能多，也不能少。*/
    rlen = abcdk_buffer_read(event->buf,(char*)event->name,ABCDK_MIN(et.len,PATH_MAX));
    if (rlen <= 0 || rlen != et.len)
        return -1;

    event->wd = et.wd;
    event->mask = et.mask;
    event->cookie = et.cookie;

    return 0;
}