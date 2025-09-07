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
NATIVE_PLATFORM=$(${SHELLDIR}/get-compiler-platform.sh "${1}")
checkReturnCode

#
NATIVE_PLATFORM=$(${SHELLDIR}/get-compiler-platform.sh "${1}")
checkReturnCode

#
NATIVE_VERSION=$(${SHELLDIR}/get-compiler-version.sh "${1}")
checkReturnCode

#
TARGET_PLATFORM=$(${SHELLDIR}/get-compiler-platform.sh "${2}")
checkReturnCode

#
TARGET_PLATFORM=$(${SHELLDIR}/get-compiler-platform.sh "${2}")
checkReturnCode

#
TARGET_SYSROOT=$(${SHELLDIR}/get-compiler-sysroot.sh "${2}")
checkReturnCode

#
TARGET_VERSION=$(${SHELLDIR}/get-compiler-version.sh "${2}")
checkReturnCode

#
TARGET_BITWIDE=$(${SHELLDIR}/get-compiler-bitwide.sh "${2}")
checkReturnCode

#提取目标平台的glibc最大版本。
if [ -L ${TARGET_SYSROOT}/lib${TARGET_BITWIDE}/libc.so.6 ];then
    TARGET_GLIBC_MAX_VERSION=$(basename $(readlink ${TARGET_SYSROOT}/lib${TARGET_BITWIDE}/libc.so.6) |grep -o 'libc-[0-9]\+\.[0-9]\+' | cut -d '-' -f2)
elif [ -L ${TARGET_SYSROOT}/lib/libc.so.6 ];then
    TARGET_GLIBC_MAX_VERSION=$(basename $(readlink ${TARGET_SYSROOT}/lib/libc.so.6) |grep -o 'libc-[0-9]\+\.[0-9]\+' | cut -d '-' -f2)
elif [ "${NATIVE_PLATFORM}" == "${TARGET_PLATFORM}" ] && [ "${NATIVE_VERSION}" == "${TARGET_VERSION}" ];then
{
    if [ -L /usr/lib${TARGET_BITWIDE}/libc.so.6 ];then
        TARGET_GLIBC_MAX_VERSION=$(basename $(readlink /usr/lib${TARGET_BITWIDE}/libc.so.6) |grep -o 'libc-[0-9]\+\.[0-9]\+' | cut -d '-' -f2)
    elif [ -L /usr/lib/${TARGET_PLATFORM}-linux-gnu/libc.so.6 ];then
        TARGET_GLIBC_MAX_VERSION=$(basename $(readlink /usr/lib/${TARGET_PLATFORM}-linux-gnu/libc.so.6) |grep -o 'libc-[0-9]\+\.[0-9]\+' | cut -d '-' -f2)
    elif  [ -L /usr/lib/libc.so.6 ];then
        TARGET_GLIBC_MAX_VERSION=$(basename $(readlink /usr/lib/libc.so.6) |grep -o 'libc-[0-9]\+\.[0-9]\+' | cut -d '-' -f2)
    else 
        TARGET_GLIBC_MAX_VERSION=$(ldd --version |head -n 1 |rev |cut -d ' ' -f 1 |rev)
    fi
}
else
    TARGET_GLIBC_MAX_VERSION="0.0"
fi

echo "${TARGET_GLIBC_MAX_VERSION}"