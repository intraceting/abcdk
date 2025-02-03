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
PKG_MACHINE=${_3RDPARTY_PKG_MACHINE}
PKG_WORDBIT=${_3RDPARTY_PKG_WORDBIT}
PKG_FIND_ROOT=${_3RDPARTY_PKG_FIND_ROOT}
PKG_FIND_MODE=${_3RDPARTY_PKG_FIND_MODE}

#修复默认值。
if [ "${PKG_FIND_ROOT}" == "" ];then
PKG_FIND_MODE="default"
fi

#
if [ "${PKG_FIND_MODE}" == "only" ];then
{
    if [ -f "${PKG_FIND_ROOT}/include/${HDNAME}" ];then
        echo "${PKG_FIND_ROOT}/include/"
    elif [ -f "${PKG_FIND_ROOT}/${HDNAME}" ];then
        echo "${PKG_FIND_ROOT}/"
    else 
        exit 1
    fi  
}
elif [ "${PKG_FIND_MODE}" == "both" ];then
{
    if [ -f "${PKG_FIND_ROOT}/include/${HDNAME}" ];then
        echo "${PKG_FIND_ROOT}/include/"
    elif [ -f "${PKG_FIND_ROOT}/${HDNAME}" ];then
        echo "${PKG_FIND_ROOT}/"
    elif [ -f "/usr/include/${HDNAME}" ];then
        echo "/usr/include/"
    elif [ -f "/usr/${HDNAME}" ];then
        echo "/usr/"
    else 
        exit 1
    fi  
}
else
{
    if [ -f "/usr/include/${HDNAME}" ];then
        echo "/usr/include/"
    elif [ -f "/usr/${HDNAME}" ];then
        echo "/usr/"
    else 
        exit 1
    fi
}
fi

#
exit $?