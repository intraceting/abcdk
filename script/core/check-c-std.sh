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
${COMPILER} -std=${STD} -x c -fsyntax-only ${SHELLDIR}/test-c-std/sample.c >/dev/null 2>&1
exit $?
