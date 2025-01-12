#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
##
#
SHELLDIR=$(cd `dirname $0`; pwd)

#检查screen是否已安装。
STATUS=$(${SHELLDIR}/check-which.sh screen)
if [ ${STATUS} -ne 0 ];then
{
    exit 1
}
fi

#
if [ $# -lt 1 ];then
{
    exit 22
}
fi

#命令行
CMDLINE=$*
#命令名字
CMDNAME=$(basename $1)

#交互式启动。
#screen -x -S ${CMDNAME} -p 0 -X stuff "${CMDLINE}"
#screen -x -S ${CMDNAME} -p 0 -X stuff $'\n'

#直接启动。
screen -d -m -S ${CMDNAME} ${CMDLINE}
