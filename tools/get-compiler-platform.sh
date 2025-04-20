#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
# 
##
#
SHELLDIR=$(cd `dirname $0`; pwd)

# Functions
checkReturnCode()
{
    rc=$?
    if [ $rc != 0 ];then
        exit $rc
    fi
}

#
TARGET_MACHINE=$(${SHELLDIR}/get-compiler-machine.sh "${1}")
checkReturnCode

#提取第一段当平台。
TARGET_PLATFORM=$(echo ${TARGET_MACHINE} | cut -d - -f 1)

#
if [ "${TARGET_PLATFORM}" == "" ];then
exit 127
fi

echo "${TARGET_PLATFORM}"