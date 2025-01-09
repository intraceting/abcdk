#!/bin/bash
#
# This file is part of SHORTCUT.
#  
# Copyright (c) 2025 The SHORTCUT project authors. All Rights Reserved.
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 2 ];then
    echo "0"
    exit 22
fi

#
NUM=$(echo "$1" |grep -wi "$2" | wc -l)
echo ${NUM}

#
exit 0