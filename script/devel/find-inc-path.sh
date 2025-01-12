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
if [ "${_THIRDPARTY_PREFIX}" == "" ];then
_THIRDPARTY_PREFIX="/usr/"
fi

#
if [ -f ${_THIRDPARTY_PREFIX}/include/${HDNAME} ];then
    echo "${_THIRDPARTY_PREFIX}/include/"
elif [ -f ${_THIRDPARTY_PREFIX}/${HDNAME} ];then
    echo "${_THIRDPARTY_PREFIX}/"
else 
    exit 1
fi

#
exit $?