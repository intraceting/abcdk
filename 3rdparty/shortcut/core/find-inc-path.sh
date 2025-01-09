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

#
HDNAME="$1"

#
if [ "${SHORTCUT_PKG_PREFIX}" == "" ];then
SHORTCUT_PKG_PREFIX="/usr/"
fi

#
if [ -f ${SHORTCUT_PKG_PREFIX}/include/${HDNAME} ];then
    echo "${SHORTCUT_PKG_PREFIX}/include/"
elif [ -f ${SHORTCUT_PKG_PREFIX}/${HDNAME} ];then
    echo "${SHORTCUT_PKG_PREFIX}/"
else 
    exit 1
fi

#
exit $?