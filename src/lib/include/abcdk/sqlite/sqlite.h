/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_SQLITE_SQLITE_H
#define ABCDK_SQLITE_SQLITE_H

#include "abcdk/util/general.h"

#ifdef HAVE_SQLITE
#include <sqlite3.h>
#endif //HAVE_SQLITE

#if !defined(_SQLITE3_H_) && !defined(SQLITE3_H)
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
#define SQLITE_OK 0
#define SQLITE_ERROR 1
#endif // #if !defined(_SQLITE3_H_) && !defined(SQLITE3_H)

#endif //ABCDK_SQLITE_SQLITE_H