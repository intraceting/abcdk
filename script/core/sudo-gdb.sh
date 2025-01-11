#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
#
SHELLDIR=$(cd `dirname $0`; pwd)

#如果未指定调试器的前缀，则使用本机默认的。
if [ "${ABCDK_THIRDPARTY_PREFIX}" == "" ];then
ABCDK_THIRDPARTY_PREFIX=/usr/bin/
fi 

#启动调试器。
pkexec --user root ${ABCDK_THIRDPARTY_PREFIX}/gdb "$@"


#在VSCODE环境中，把当前脚本的路径作为"miDebuggerPath"字段的值。
#例："miDebuggerPath": "${workspaceFolder}/script/devel/sudo-gdb.sh",