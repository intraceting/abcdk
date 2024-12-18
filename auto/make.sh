#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
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
BUILD_PATH=${SHELLDIR}/../build/auto.build/
#
WORK_SPACE=${SHELLDIR}/../

#
make -s -C ${WORK_SPACE} MAKE_CONF=${BUILD_PATH}/makefile.conf $*
exit_if_error $? "make failed." 1
