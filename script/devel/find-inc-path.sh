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

#
HDNAME="$1"

#
PKG_PREFIX=${_THIRDPARTY_PKG_PREFIX}
PKG_FIND_MODE=${_THIRDPARTY_PKG_FIND_MODE}

#修复默认值。
if [ "${PKG_PREFIX}" == "" ];then
PKG_FIND_MODE="default"
fi

#
if [ "${PKG_FIND_MODE}" == "only" ];then
{
    if [ -f ${PKG_PREFIX}/include/${HDNAME} ];then
        echo "${PKG_PREFIX}/include/"
    elif [ -f ${PKG_PREFIX}/${HDNAME} ];then
        echo "${PKG_PREFIX}/"
    else 
        exit 1
    fi  
}
elif [ "${PKG_FIND_MODE}" == "both" ];then
{
    if [ -f ${PKG_PREFIX}/include/${HDNAME} ];then
        echo "${PKG_PREFIX}/include/"
    elif [ -f ${PKG_PREFIX}/${HDNAME} ];then
        echo "${PKG_PREFIX}/"
    elif [ -f /usr/include/${HDNAME} ];then
        echo "/usr/include/"
    elif [ -f /usr/${HDNAME} ];then
        echo "/usr/"
    else 
        exit 1
    fi  
}
else
{
    if [ -f /usr/include/${HDNAME} ];then
        echo "/usr/include/"
    elif [ -f /usr/${HDNAME} ];then
        echo "/usr/"
    else 
        exit 1
    fi
}
fi

#
exit $?