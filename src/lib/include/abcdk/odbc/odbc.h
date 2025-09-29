/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_ODBC_ODBC_H
#define ABCDK_ODBC_ODBC_H

#include "abcdk/util/general.h"

#ifdef HAVE_UNIXODBC
#include <sql.h>
#include <sqlext.h>
#endif // HAVE_UNIXODBC

#ifndef __SQL_H
#define SQL_SUCCESS 0
#define SQL_ERROR (-1)
#endif // #ifndef __SQL_H

#ifndef __SQLTYPES_H
typedef signed short int   SQLSMALLINT;
typedef unsigned short  SQLUSMALLINT;
typedef SQLSMALLINT     SQLRETURN;
typedef void *          SQLPOINTER;
typedef unsigned char   SQLCHAR;

#if (SIZEOF_LONG_INT == 8)

#ifdef BUILD_LEGACY_64_BIT_MODE

typedef int             SQLINTEGER;
typedef unsigned int    SQLUINTEGER;
#define SQLLEN          SQLINTEGER
#define SQLULEN         SQLUINTEGER
#define SQLSETPOSIROW   SQLUSMALLINT

#else //#ifdef BUILD_LEGACY_64_BIT_MODE

typedef int             SQLINTEGER;
typedef unsigned int    SQLUINTEGER;
typedef long            SQLLEN;
typedef unsigned long   SQLULEN;
typedef unsigned long   SQLSETPOSIROW;

#endif //#ifdef BUILD_LEGACY_64_BIT_MODE

#else //#if (SIZEOF_LONG_INT == 8)

typedef long            SQLINTEGER;
typedef unsigned long   SQLUINTEGER;

#ifdef _WIN64
typedef long long SQLLEN;
typedef unsigned long long SQLULEN;
typedef unsigned long long SQLSETPOSIROW;
#else //#ifdef _WIN64
#define SQLLEN          SQLINTEGER
#define SQLULEN         SQLUINTEGER
#define SQLSETPOSIROW   SQLUSMALLINT
#endif //#ifdef _WIN64

typedef SQLULEN         SQLROWCOUNT;
typedef SQLULEN         SQLROWSETSIZE;
typedef SQLULEN         SQLTRANSID;
typedef SQLLEN          SQLROWOFFSET;

#endif //#if (SIZEOF_LONG_INT == 8)

#if (ODBCVER >= 0x0300)
typedef void * 			        SQLHANDLE; 
typedef SQLHANDLE               SQLHENV;
typedef SQLHANDLE               SQLHDBC;
typedef SQLHANDLE               SQLHSTMT;
typedef SQLHANDLE               SQLHDESC;
#else //#if (ODBCVER >= 0x0300)
typedef void *                  SQLHENV;
typedef void *                  SQLHDBC;
typedef void *                  SQLHSTMT;
typedef void * 			        SQLHANDLE; 
#endif //#if (ODBCVER >= 0x0300)

#endif //#ifndef __SQLTYPES_H

#endif //ABCDK_ODBC_ODBC_H