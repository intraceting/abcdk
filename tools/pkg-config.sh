#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

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
PKG_CFG_PATH=${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/${PKG_MACHINE}/pkgconfig:${PKG_FIND_ROOT}/lib${PKG_WORDBIT}/pkgconfig:${PKG_FIND_ROOT}/lib/${PKG_MACHINE}/pkgconfig:${PKG_FIND_ROOT}/lib/pkgconfig:${PKG_FIND_ROOT}/share/pkgconfig



#
if [ "${PKG_FIND_MODE}" == "only" ];then
export PKG_CONFIG_LIBDIR=${PKG_CFG_PATH}
elif [ "${PKG_FIND_MODE}" == "both" ];then
export PKG_CONFIG_PATH=${PKG_CFG_PATH}
fi

#
pkg-config $@ 2>>/dev/null

