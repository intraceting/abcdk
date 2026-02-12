/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#ifndef ABCDK_REDIS_REDIS_H
#define ABCDK_REDIS_REDIS_H

#include "abcdk/util/general.h"

#ifdef HAVE_HIREDIS
#include <hiredis/hiredis.h>
#endif //HAVE_HIREDIS

#ifndef __HIREDIS_H
typedef struct redisReply redisReply;
typedef struct redisContext redisContext;
#endif //#ifndef __HIREDIS_H

#endif //ABCDK_REDIS_REDIS_H