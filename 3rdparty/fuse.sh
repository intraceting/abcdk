#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
##

#
SHELLDIR=$(cd `dirname "$0"`; pwd)

# Functions
checkReturnCode()
{
    rc=$?
    if [ $rc != 0 ];then
        exit $rc
    fi
}


#
CheckSystemName()
# $1 System Name
{
    ${SHELLKITS_HOME}/tools/check-os-id.sh "$1"
}

#
GetSystemVersion()
{
    ${SHELLKITS_HOME}/tools/get-os-ver.sh
}

#
CheckPackageKitName()
{
	${SHELLKITS_HOME}/tools/get-kit-name.sh
}

#
CheckHavePackageFromKit()
# $1 PACKAGE
{
    ${SHELLKITS_HOME}/tools/check-package.sh "$1"
}

#
CheckHavePackageFromWhich()
# $1 PACKAGE
{
	${SHELLKITS_HOME}/tools/check-which.sh "$1"
}

#
FindIncPath()
# $1 HDNAME
{
	${SHELLKITS_HOME}/tools/find-inc-path.sh "$1"
}

#
FindLibPath()
# $1 SONAME
{
	${SHELLKITS_HOME}/tools/find-lib-path.sh "$1"
}

#
PackageConfig()
# $1 SONAME
{
	${SHELLKITS_HOME}/tools/pkg-config.sh $@
}


#
if [ $# -lt 1 ];then
exit 22
fi

#
SYS_VERID=$(GetSystemVersion)
KIT_NAME=$(CheckPackageKitName)
FLAG="$1"
   
#
if [ "deb" == "${KIT_NAME}" ];then 
{ 
    if [ ${FLAG} -eq 1 ];then
        exit $(CheckHavePackageFromKit "libfuse-dev")
    elif [ ${FLAG} -eq 2 ];then
        PackageConfig  --cflags fuse 
    elif [ ${FLAG} -eq 3 ];then
        PackageConfig  --libs fuse 
    elif [ ${FLAG} -eq 4 ];then
        echo "libfuse-dev"
    else
        exit 22
    fi
}
elif [ "rpm" == "${KIT_NAME}" ];then 
{
    if [ ${FLAG} -eq 1 ];then
        exit $(CheckHavePackageFromKit "fuse-devel")
    elif [ ${FLAG} -eq 2 ];then
        PackageConfig  --cflags fuse 
    elif [ ${FLAG} -eq 3 ];then
        PackageConfig  --libs fuse 
    elif [ ${FLAG} -eq 4 ];then
        echo "fuse-devel"
    else
        exit 22
    fi
}
else
{
    exit 1
}
fi 

#
exit $?
