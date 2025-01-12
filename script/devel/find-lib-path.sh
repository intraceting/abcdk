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
if [ "${_THIRDPARTY_PREFIX}" == "" ];then
_THIRDPARTY_PREFIX="/usr/"
fi

#
if [ "${_THIRDPARTY_MACHINE}" == "" ];then
_THIRDPARTY_MACHINE="$(uname -m)-linux-gnu"
fi 

#
if [ "${_THIRDPARTY_BITWIDE}" == "" ];then
{
    if [ "$(getconf WORD_BIT)" == "32" ] && [ "$(getconf LONG_BIT)" == "64" ];then
        _THIRDPARTY_BITWIDE="64"
    else 
        _THIRDPARTY_BITWIDE="32"
    fi 
}
fi

#
if [ -f ${_THIRDPARTY_PREFIX}/lib${_THIRDPARTY_BITWIDE}/${_THIRDPARTY_MACHINE}/${SONAME} ];then
    echo "${_THIRDPARTY_PREFIX}/lib${_THIRDPARTY_BITWIDE}/${_THIRDPARTY_MACHINE}/"
elif [ -f ${_THIRDPARTY_PREFIX}/lib/${_THIRDPARTY_MACHINE}/${SONAME} ];then
    echo "${_THIRDPARTY_PREFIX}/lib/${_THIRDPARTY_MACHINE}/"
elif [ -f ${_THIRDPARTY_PREFIX}/lib${_THIRDPARTY_BITWIDE}/${SONAME} ];then
    echo "${_THIRDPARTY_PREFIX}/lib${_THIRDPARTY_BITWIDE}/"
elif [ -f ${_THIRDPARTY_PREFIX}/lib/${SONAME} ];then
    echo "${_THIRDPARTY_PREFIX}/lib/"
else 
    exit 1
fi

#
exit $?