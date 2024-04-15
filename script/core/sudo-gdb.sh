#!/bin/bash -x
#
# This file is part of ABCDK.
#  
# MIT License
#
SHELLDIR=$(cd `dirname $0`; pwd)

#如果未指定调试器的前缀，则使用本机默认的。
if [ "${GDB_PREFIX}" == "" ];then
GDB_PREFIX=/usr/bin/
fi 

#启动调试器。
pkexec --user root ${GDB_PREFIX}/gdb "$@"