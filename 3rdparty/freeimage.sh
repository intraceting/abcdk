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
    ${SHELLDIR}/../tools/check-os-id.sh "$1"
}

#
GetSystemVersion()
{
    ${SHELLDIR}/../tools/get-os-ver.sh
}

#
CheckPackageKitName()
{
	${SHELLDIR}/../tools/get-kit-name.sh
}

#
CheckHavePackageFromKit()
# $1 PACKAGE
{
    ${SHELLDIR}/../tools/check-package.sh "$1"
}

#
CheckHavePackageFromWhich()
# $1 PACKAGE
{
	${SHELLDIR}/../tools/check-which.sh "$1"
}

#
FindIncPath()
# $1 HDNAME
{
	${SHELLDIR}/../tools/find-inc-path.sh "$1"
}

#
FindLibPath()
# $1 SONAME
{
	${SHELLDIR}/../tools/find-lib-path.sh "$1"
}

#
PackageConfig()
# $1 SONAME
{
	${SHELLDIR}/../tools/pkg-config.sh $@
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
        exit $(CheckHavePackageFromKit "libfreeimage-dev")
    elif [ ${FLAG} -eq 2 ];then
    {
        CFLAG="-I$(FindIncPath FreeImage.h)"
        checkReturnCode

        echo "${CFLAG}"
    }
    elif [ ${FLAG} -eq 3 ];then
    {
        LDFLAG="-L$(FindLibPath libfreeimage.so)"
        checkReturnCode

        echo "-lfreeimage ${LDFLAG}"
    }
    elif [ ${FLAG} -eq 4 ];then
        echo "libfreeimage-dev"
    else
        exit 22
    fi
}
elif [ "rpm" == "${KIT_NAME}" ];then 
{
    if [ ${FLAG} -eq 1 ];then
        exit $(CheckHavePackageFromKit "freeimage-devel")
    elif [ ${FLAG} -eq 2 ];then
    {
        CFLAG="-I$(FindIncPath FreeImage.h)"
        checkReturnCode

        echo "${CFLAG}"
    }
    elif [ ${FLAG} -eq 3 ];then
    {
        LDFLAG="-L$(FindLibPath libfreeimage.so)"
        checkReturnCode

        echo "-lfreeimage ${LDFLAG}"
    }
    elif [ ${FLAG} -eq 4 ];then
        echo "freeimage-devel"
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
