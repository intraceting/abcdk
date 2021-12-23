/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_COMM_WAITER_H
#define ABCDK_COMM_WAITER_H

#include "abcdk-comm/comm.h"
#include "abcdk-comm/message.h"

__BEGIN_DECLS

typedef struct _abcdk_comm_waiter
{


} abcdk_comm_waiter_t;

/** */
typedef int (*abcdk_comm_waiter_cb)(const void *data,size_t size,void *opaque);

abcdk_comm_waiter_t *abcdk_comm_waiter_alloc();

__END_DECLS


#endif //ABCDK_COMM_WAITER_H