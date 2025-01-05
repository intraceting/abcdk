#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
##


#
exit_if_error()
#errno
#errstr
#exitcode
{
    if [ $# -ne 3 ];then
    {
        echo "需要三个参数，分别是：errno，errstr，exitcode。"
        exit 1
    }
    fi 
    
    if [ $1 -ne 0 ];then
    {
        echo $2
        exit $3
    }
    fi
}

#
SHELLDIR=$(cd `dirname "$0"`; pwd)

#
${SHELLDIR}/configure.sh -O -o 3 $*
exit_if_error $? "configure failed." 1

#
${SHELLDIR}/make.sh clean
exit_if_error $? "make failed." 1

#
${SHELLDIR}/make.sh -j4
exit_if_error $? "make failed." 1

#
${SHELLDIR}/make.sh package
exit_if_error $? "make failed." 1