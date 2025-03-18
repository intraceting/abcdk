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
SONAME="$1"

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
if [ "${PKG_MACHINE}" == "" ];then
PKG_MACHINE="$(uname -m)-linux-gnu"
fi 

#
if [ "${PKG_WORDBIT}" == "" ];then
{
    if [ "$(getconf WORD_BIT)" == "32" ] && [ "$(getconf LONG_BIT)" == "64" ];then
        PKG_WORDBIT="64"
    else 
        PKG_WORDBIT="32"
    fi 
}
fi

#
if [ "${PKG_FIND_MODE}" == "only" ];then
{
    if [ -f "${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/${PKG_MACHINE}/${SONAME}" ];then
        echo "${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/${PKG_MACHINE}/"
    elif [ -f "${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/${SONAME}" ];then
        echo "${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/"
    elif [ -f "${PKG_FIND_ROOT}/lib/${PKG_MACHINE}/${SONAME}" ];then
        echo "${PKG_FIND_ROOT}/lib/${PKG_MACHINE}/"
    elif [ -f "${PKG_FIND_ROOT}/lib/${SONAME}" ];then
        echo "${PKG_FIND_ROOT}/lib/"
    else 
        exit 1
    fi
}
elif [ "${PKG_FIND_MODE}" == "both" ];then
{
    if [ -f "${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/${PKG_MACHINE}/${SONAME}" ];then
        echo "${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/${PKG_MACHINE}/"
    elif [ -f "${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/${SONAME}" ];then
        echo "${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/"
    elif [ -f "${PKG_FIND_ROOT}/lib/${PKG_MACHINE}/${SONAME}" ];then
        echo "${PKG_FIND_ROOT}/lib/${PKG_MACHINE}/"
    elif [ -f "${PKG_FIND_ROOT}/lib/${SONAME}" ];then
        echo "${PKG_FIND_ROOT}/lib/"
    elif [ -f "/usr/lib${PKG_WORDBIT}/${PKG_MACHINE}/${SONAME}" ];then
        echo "/usr/lib${PKG_WORDBIT}/${PKG_MACHINE}/"
    elif [ -f "/usr/lib${PKG_WORDBIT}/${SONAME}" ];then
        echo "/usr/lib${PKG_WORDBIT}/"
    elif [ -f "/usr/lib/${PKG_MACHINE}/${SONAME}" ];then
        echo "/usr/lib/${PKG_MACHINE}/"
    elif [ -f "/usr/lib/${SONAME}" ];then
        echo "/usr/lib/"
    else
        exit 1
    fi
}
else
{
    if [ -f "/usr/lib${PKG_WORDBIT}/${PKG_MACHINE}/${SONAME}" ];then
        echo "/usr/lib${PKG_WORDBIT}/${PKG_MACHINE}/"
    elif [ -f "/usr/lib${PKG_WORDBIT}/${SONAME}" ];then
        echo "/usr/lib${PKG_WORDBIT}/"
    elif [ -f "/usr/lib/${PKG_MACHINE}/${SONAME}" ];then
        echo "/usr/lib/${PKG_MACHINE}/"
    elif [ -f "/usr/lib/${SONAME}" ];then
        echo "/usr/lib/"
    else
        exit 1
    fi
}
fi

#
exit $?