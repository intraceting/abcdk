#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) intraceting<intraceting@outlook.com>
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