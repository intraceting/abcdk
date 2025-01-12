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
PKG_TARGET_MACHINE=${_PKG_TARGET_MACHINE}
PKG_TARGET_WORDBIT=${_PKG_TARGET_WORDBIT}
PKG_PREFIX=${_THIRDPARTY_PKG_PREFIX}
PKG_FIND_MODE=${_THIRDPARTY_PKG_FIND_MODE}

#修复默认值。
if [ "${PKG_PREFIX}" == "" ];then
PKG_FIND_MODE="default"
fi

#
if [ "${PKG_TARGET_MACHINE}" == "" ];then
PKG_TARGET_MACHINE="$(uname -m)-linux-gnu"
fi 

#
if [ "${PKG_TARGET_WORDBIT}" == "" ];then
{
    if [ "$(getconf WORD_BIT)" == "32" ] && [ "$(getconf LONG_BIT)" == "64" ];then
        PKG_TARGET_WORDBIT="64"
    else 
        PKG_TARGET_WORDBIT="32"
    fi 
}
fi

#
if [ "${PKG_FIND_MODE}" == "only" ];then
{
    if [ -f ${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/${SONAME} ];then
        echo "${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/"
    elif [ -f ${PKG_PREFIX}/lib/${PKG_TARGET_MACHINE}/${SONAME} ];then
        echo "${PKG_PREFIX}/lib/${PKG_TARGET_MACHINE}/"
    elif [ -f ${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/${SONAME} ];then
        echo "${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/"
    elif [ -f ${PKG_PREFIX}/lib/${SONAME} ];then
        echo "${PKG_PREFIX}/lib/"
    else 
        exit 1
    fi
}
elif [ "${PKG_FIND_MODE}" == "both" ];then
{
    if [ -f ${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/${SONAME} ];then
        echo "${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/"
    elif [ -f ${PKG_PREFIX}/lib/${PKG_TARGET_MACHINE}/${SONAME} ];then
        echo "${PKG_PREFIX}/lib/${PKG_TARGET_MACHINE}/"
    elif [ -f ${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/${SONAME} ];then
        echo "${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/"
    elif [ -f ${PKG_PREFIX}/lib/${SONAME} ];then
        echo "${PKG_PREFIX}/lib/"
    elif [ -f /usr/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/${SONAME} ];then
        echo "/usr/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/"
    elif [ -f /usr/lib/${PKG_TARGET_MACHINE}/${SONAME} ];then
        echo "/usr/lib/${PKG_TARGET_MACHINE}/"
    elif [ -f /usr/lib${PKG_TARGET_WORDBIT}/${SONAME} ];then
        echo "/usr/lib${PKG_TARGET_WORDBIT}/"
    elif [ -f /usr/lib/${SONAME} ];then
        echo "/usr/lib/"
    else
        exit 1
    fi
}
else
{
    if [ -f /usr/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/${SONAME} ];then
        echo "/usr/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/"
    elif [ -f /usr/lib/${PKG_TARGET_MACHINE}/${SONAME} ];then
        echo "/usr/lib/${PKG_TARGET_MACHINE}/"
    elif [ -f /usr/lib${PKG_TARGET_WORDBIT}/${SONAME} ];then
        echo "/usr/lib${PKG_TARGET_WORDBIT}/"
    elif [ -f /usr/lib/${SONAME} ];then
        echo "/usr/lib/"
    else
        exit 1
    fi
}
fi

#
exit $?