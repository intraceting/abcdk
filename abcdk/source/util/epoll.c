/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/util/epoll.h"

int abcdk_epoll_create()
{
#if 0
    int fd = epoll_create(1024);
#else 
    int fd = epoll_create1(0);
#endif 
    if (fd < 0)
        return -1;

    /* 添加个非必要标志，忽略可能的出错信息。 */
    abcdk_fflag_add(fd, SOCK_CLOEXEC);

    return fd;
}

int abcdk_epoll_mark(int efd, int fd, const abcdk_epoll_event_t *event, int first)
{
    int opt = 0;
    abcdk_epoll_event_t mark;

    assert(efd >= 0 && fd >= 0 && event != NULL);
    assert((event->events & ~(ABCDK_EPOLL_INPUT | ABCDK_EPOLL_OUTPUT | ABCDK_EPOLL_ERROR)) == 0);

    memset(&mark,0,sizeof(mark));

    /*如果注册事件中包括错误事件，则直接跳转出错流程。*/
    if (event->events & ABCDK_EPOLL_ERROR)
        goto final_error;

    mark.data = event->data;
    mark.events |= (EPOLLERR | EPOLLHUP | EPOLLRDHUP | EPOLLET);

    /*转换事件。*/
    if (event->events & ABCDK_EPOLL_INPUT)
        mark.events |= EPOLLIN;
    if (event->events & ABCDK_EPOLL_OUTPUT)
        mark.events |= EPOLLOUT;

    opt = (first ? EPOLL_CTL_ADD : EPOLL_CTL_MOD);

    return epoll_ctl(efd, opt, fd, &mark);

final_error:

    return -1;
}

int abcdk_epoll_drop(int efd, int fd)
{
    assert(efd >= 0 && fd >= 0);

    return epoll_ctl(efd,EPOLL_CTL_DEL,fd, NULL);
}

int abcdk_epoll_wait(int efd,abcdk_epoll_event_t* events,int max,time_t timeout)
{
    int chk;
    uint32_t tmp;
    
    assert(efd >= 0 && events != NULL && max > 0);

    chk = epoll_wait(efd, events, max, (timeout >= INT32_MAX ? -1 : timeout));
    if(chk == -1 && errno == EINTR)
        return 0;

    /*转换事件。 */
    for (int i = 0; i < chk; i++)
    {
        tmp = events[i].events;
        events[i].events = 0;

        if(tmp & EPOLLIN)
            events[i].events |= ABCDK_EPOLL_INPUT;
        if(tmp & EPOLLOUT)
            events[i].events |= ABCDK_EPOLL_OUTPUT;
        if(tmp & (EPOLLERR | EPOLLHUP | EPOLLRDHUP))
            events[i].events |= ABCDK_EPOLL_ERROR;
    }

    return chk;
}