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

#转换成绝对路径。
COMPILER_BIN=$(which "${1}")

#
TARGET_PROG_NAME=$(${COMPILER_BIN} "-print-prog-name=${2}" 2>>/dev/null)
checkReturnCode

#
TARGET_PROG_NAME=$(which "${TARGET_PROG_NAME}")
checkReturnCode

#
echo "${TARGET_PROG_NAME}"