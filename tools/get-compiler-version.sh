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

#在GCC7(包括)之后, 参数'-dumpversion'默认只显示主版本. 如果需要显示完整版本号可以使用-dumpfullversion参数.
#GCC组织承诺, 在GCC7(包括)之后发布的版本, 当主版号相同时会保持ABI稳定, 副版本号及修正版本号的变化不影向兼容性.

#
VERSION_STR=$(${COMPILER_BIN} "-dumpversion" 2>>/dev/null)
checkReturnCode

#
if [ "${VERSION_STR}" == "" ];then
exit 1
fi

#
VERSION_MAJOR=$(echo ${VERSION_STR} | cut -d '.' -f 1)
if [ "${VERSION_MAJOR}" -ge 7 ]; then
    echo "${VERSION_MAJOR}"
else 
    echo "${TARGET_VERSION}"
fi

#
exit 0
