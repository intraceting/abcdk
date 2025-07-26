#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
# 
##
#
SHELLNAME=$(basename ${0})
SHELLDIR=$(cd `dirname ${0}`; pwd)

#导出必要的环境变量。
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SHELLDIR}:${SHELLDIR}/../lib64:${SHELLDIR}/../lib64/compat:${SHELLDIR}/../lib:${SHELLDIR}/../lib/compat

#启动可执行程序。
${0}.exe $@
exit $?
