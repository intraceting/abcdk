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

#兼容性写法，因为在gcc7之后参数'-dumpversion'默认只显示主版本。
TARGET_VERSION=$(${COMPILER_BIN} "-dumpfullversion" 2>>/dev/null)
if [ $? -ne 0 ] || [ "${TARGET_VERSION}" == "" ];then
TARGET_VERSION=$(${COMPILER_BIN} "-dumpversion" 2>>/dev/null)
checkReturnCode
fi


#
echo "${TARGET_VERSION}"