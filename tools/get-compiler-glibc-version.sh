#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
# 
##
#
SHELLDIR=$(cd `dirname $0`; pwd)

# Functions
checkReturnCode()
{
    rc=$?
    if [ $rc != 0 ];then
        exit $rc
    fi
}

#
NATIVE_PLATFORM=$(${SHELLDIR}/get-compiler-platform.sh "gcc")
checkReturnCode

#
TARGET_PLATFORM=$(${SHELLDIR}/get-compiler-platform.sh "${1}")
checkReturnCode

#
TARGET_SYSROOT=$(${SHELLDIR}/get-compiler-sysroot.sh "${1}")
checkReturnCode

#
TARGET_READELF=$(${SHELLDIR}/get-compiler-prog-name.sh "${1}" "readelf")
checkReturnCode

#
TARGET_BITWIDE=$(${SHELLDIR}/get-compiler-bitwide.sh "${1}")
checkReturnCode


#
if [ "${NATIVE_PLATFORM}" == "${TARGET_PLATFORM}" ];then
{
    #提取目标平台的glibc最大版本。
    TARGET_GLIBC_MAX_VER=$(ldd --version |head -n 1 |rev |cut -d ' ' -f 1 |rev)
}
else
{
    #提取目标平台的glibc最大版本。
    if [ -f ${TARGET_SYSROOT}/lib${TARGET_BITWIDE}/libc.so.6 ];then
        TARGET_GLIBC_MAX_VER=$(${TARGET_READELF} -V ${TARGET_SYSROOT}/lib${TARGET_BITWIDE}/libc.so.6 | grep -o 'GLIBC_[0-9]\+\.[0-9]\+' | sort -u -V -r |head -n 1 |cut -d '_' -f 2)
    elif [ -f ${TARGET_SYSROOT}/lib/libc.so.6 ];then
        TARGET_GLIBC_MAX_VER=$(${TARGET_READELF} -V ${TARGET_SYSROOT}/lib/libc.so.6 | grep -o 'GLIBC_[0-9]\+\.[0-9]\+' | sort -u -V -r |head -n 1 |cut -d '_' -f 2)
    elif [ -f ${TARGET_SYSROOT}/${TARGET_PLATFORM}-linux-gnu/lib/libc.so.6 ];then
        TARGET_GLIBC_MAX_VER=$(${TARGET_READELF} -V ${TARGET_SYSROOT}/lib/libc.so.6 | grep -o 'GLIBC_[0-9]\+\.[0-9]\+' | sort -u -V -r |head -n 1 |cut -d '_' -f 2)
    else
        TARGET_GLIBC_MAX_VER="0.0"
    fi
}
fi

echo "${TARGET_GLIBC_MAX_VER}"