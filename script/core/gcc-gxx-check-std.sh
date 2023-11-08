#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
##
#
SHELLDIR=$(cd `dirname $0`; pwd)


#
if [ $# -ne 2 ];then
{
    exit 22
}
fi

#
COMPILER=$1
STD=$2

#
${COMPILER} -E -dM -std=${STD} - </dev/null >/dev/null 2>&1
exit $?
