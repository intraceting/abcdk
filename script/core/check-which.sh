#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
##
#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 1 ];then
{
    exit 22
}
fi

#0 已安装，!0 未安装。
STATUS="1"

#
STATUS=$(which ${1} >> /dev/null 2>&1 ; echo $?)

#
exit ${STATUS}