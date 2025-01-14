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
    ${SHELLDIR}/../script/devel/check-os-id.sh "$1"
}

#
GetSystemVersion()
{
    ${SHELLDIR}/../script/devel/get-os-ver.sh
}

#
CheckPackageKitName()
{
	${SHELLDIR}/../script/devel/get-kit-name.sh
}

#
CheckHavePackageFromKit()
# $1 PACKAGE
{
    ${SHELLDIR}/../script/devel/check-package.sh "$1"
}

#
CheckHavePackageFromWhich()
# $1 PACKAGE
{
	${SHELLDIR}/../script/devel/check-which.sh "$1"
}

#
FindIncPath()
# $1 HDNAME
{
	${SHELLDIR}/../script/devel/find-inc-path.sh "$1"
}

#
FindLibPath()
# $1 SONAME
{
	${SHELLDIR}/../script/devel/find-lib-path.sh "$1"
}

#
PackageConfig()
# $1 SONAME
{
	${SHELLDIR}/../script/devel/pkg-config.sh $@
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
    {
        A=$(CheckHavePackageFromKit dpkg | echo $?)
        B=$(CheckHavePackageFromKit dpkg-dev | echo $?)

        if [ ${A} -eq 0 ] && [ ${B} -eq 0 ];then
            exit 0
        else 
            exit 1
        fi
    } 
    elif [ ${FLAG} -eq 2 ];then
        echo ""
    elif [ ${FLAG} -eq 3 ];then
        echo ""
    elif [ ${FLAG} -eq 4 ];then
        echo "dpkg dpkg-dev"
    else
        exit 22
    fi
}
elif [ "rpm" == "${KIT_NAME}" ];then 
{
    if [ ${FLAG} -eq 1 ];then
        exit 1
    elif [ ${FLAG} -eq 2 ];then
        echo ""
    elif [ ${FLAG} -eq 3 ];then
        echo ""
    elif [ ${FLAG} -eq 4 ];then
        echo ""
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
