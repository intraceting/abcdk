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
BUILD_PATH=/tmp/abcdk.build/
#
WORK_SPACE=${SHELLDIR}/../

#
${SHELLDIR}/../configure.sh -b ${BUILD_PATH} $*
exit_if_error $? "配置错误。" 1

#
make -C ${WORK_SPACE} MAKE_CONF=${BUILD_PATH}/makefile.conf clean 
exit_if_error $? "清理错误。" 1
#

#
make -C ${WORK_SPACE} -j4 MAKE_CONF=${BUILD_PATH}/makefile.conf
exit_if_error $? "编译错误。" 1

#
make -C ${WORK_SPACE} MAKE_CONF=${BUILD_PATH}/makefile.conf package
exit_if_error $? "打包错误。" 1