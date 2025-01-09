#!/bin/bash
#
# This file is part of SHORTCUT.
#  
# Copyright (c) 2021 The SHORTCUT project authors. All Rights Reserved.
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
if [ "${SHORTCUT_PKG_PREFIX}" == "" ];then
SHORTCUT_PKG_PREFIX="/usr/"
fi

#
if [ "${SHORTCUT_PKG_PLATFORM}" == "" ];then
SHORTCUT_PKG_PLATFORM="$(uname -m)-linux-gnu"
fi 

#
if [ "${SHORTCUT_PKG_BITWIDE}" == "" ];then
{
    if [ "$(getconf WORD_BIT)" == "32" ] && [ "$(getconf LONG_BIT)" == "64" ];then
        SHORTCUT_PKG_BITWIDE="64"
    else 
        SHORTCUT_PKG_BITWIDE="32"
    fi 
}
fi

#
if [ -f ${SHORTCUT_PKG_PREFIX}/lib${SHORTCUT_PKG_BITWIDE}/${SHORTCUT_PKG_PLATFORM}/${SONAME} ];then
    echo "${SHORTCUT_PKG_PREFIX}/lib${SHORTCUT_PKG_BITWIDE}/${SHORTCUT_PKG_PLATFORM}/"
elif [ -f ${SHORTCUT_PKG_PREFIX}/lib${SHORTCUT_PKG_BITWIDE}/${SONAME} ];then
    echo "${SHORTCUT_PKG_PREFIX}/lib${SHORTCUT_PKG_BITWIDE}/"
elif [ -f ${SHORTCUT_PKG_PREFIX}/lib/${SHORTCUT_PKG_PLATFORM}/${SONAME} ];then
    echo "${SHORTCUT_PKG_PREFIX}/lib/${SHORTCUT_PKG_PLATFORM}/"
elif [ -f ${SHORTCUT_PKG_PREFIX}/lib/${SONAME} ];then
    echo "${SHORTCUT_PKG_PREFIX}/lib/"
else 
    exit 1
fi

#
exit $?