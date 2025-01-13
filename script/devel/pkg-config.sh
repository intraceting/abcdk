#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

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
PKG_CFG_PATH=${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/pkgconfig:${PKG_PREFIX}/lib${PKG_TARGET_WORDBIT}/${PKG_TARGET_MACHINE}/pkgconfig:${PKG_PREFIX}/lib/${PKG_TARGET_MACHINE}/pkgconfig:${PKG_PREFIX}/lib/pkgconfig:${PKG_PREFIX}/share/pkgconfig

#
if [ "${PKG_FIND_MODE}" == "only" ];then
{
    export PKG_CONFIG_LIBDIR=${PKG_CFG_PATH}
    pkg-config --define-variable=prefix=${PKG_PREFIX} $@
}
elif [ "${PKG_FIND_MODE}" == "both" ];then
{
    export PKG_CONFIG_PATH=${PKG_CFG_PATH}
    pkg-config $@
}
else
{
    pkg-config $@
}
fi
