#!/bin/bash
#
# This file is part of SHORTCUT.
#  
# Copyright (c) 2021 The SHORTCUT project authors. All Rights Reserved.
# 
##

#
SHELLDIR=$(cd `dirname $0`; pwd)

#
if [ $# -ne 1 ];then
{
    echo "22"
    exit 22
}
fi

#
${SHELLDIR}/get-os-id.sh | grep -iE "${1}" |wc -l