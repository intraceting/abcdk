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
if [ "${ABCDK_THIRDPARTY_PREFIX}" == "" ];then
ABCDK_THIRDPARTY_PREFIX="/usr/"
fi

#
if [ -f ${ABCDK_THIRDPARTY_PREFIX}/include/${HDNAME} ];then
    echo "${ABCDK_THIRDPARTY_PREFIX}/include/"
elif [ -f ${ABCDK_THIRDPARTY_PREFIX}/${HDNAME} ];then
    echo "${ABCDK_THIRDPARTY_PREFIX}/"
else 
    exit 1
fi

#
exit $?