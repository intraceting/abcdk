#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 1 ];then
    exit 22
fi

SONAME="$1"

#
if [ "${ABCDK_THIRDPARTY_PREFIX}" == "" ];then
ABCDK_THIRDPARTY_PREFIX="/usr/"
fi

#
if [ "${ABCDK_THIRDPARTY_MACHINE}" == "" ];then
ABCDK_THIRDPARTY_MACHINE="$(uname -m)-linux-gnu"
fi 

#
if [ "${ABCDK_THIRDPARTY_BITWIDE}" == "" ];then
{
    if [ "$(getconf WORD_BIT)" == "32" ] && [ "$(getconf LONG_BIT)" == "64" ];then
        ABCDK_THIRDPARTY_BITWIDE="64"
    else 
        ABCDK_THIRDPARTY_BITWIDE="32"
    fi 
}
fi

#
if [ -f ${ABCDK_THIRDPARTY_PREFIX}/lib${ABCDK_THIRDPARTY_BITWIDE}/${ABCDK_THIRDPARTY_MACHINE}/${SONAME} ];then
    echo "${ABCDK_THIRDPARTY_PREFIX}/lib${ABCDK_THIRDPARTY_BITWIDE}/${ABCDK_THIRDPARTY_MACHINE}/"
elif [ -f ${ABCDK_THIRDPARTY_PREFIX}/lib/${ABCDK_THIRDPARTY_MACHINE}/${SONAME} ];then
    echo "${ABCDK_THIRDPARTY_PREFIX}/lib/${ABCDK_THIRDPARTY_MACHINE}/"
elif [ -f ${ABCDK_THIRDPARTY_PREFIX}/lib${ABCDK_THIRDPARTY_BITWIDE}/${SONAME} ];then
    echo "${ABCDK_THIRDPARTY_PREFIX}/lib${ABCDK_THIRDPARTY_BITWIDE}/"
elif [ -f ${ABCDK_THIRDPARTY_PREFIX}/lib/${SONAME} ];then
    echo "${ABCDK_THIRDPARTY_PREFIX}/lib/"
else 
    exit 1
fi

#
exit $?