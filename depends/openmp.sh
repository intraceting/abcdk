#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
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
    echo "$(${SHELLDIR}/../script/core/check-os-id.sh "$1")"
}

#
GetSystemVersion()
{
    echo "$(${SHELLDIR}/../script/core/get-os-ver.sh)"
}

#
CheckPackageKitName()
{
	echo "$(${SHELLDIR}/../script/core/get-kit-name.sh)"
}

#
CheckHavePackageFromKit()
# $1 PACKAGE
{
    echo "$(${SHELLDIR}/../script/core/check-package.sh "$1")"
}

#
CheckHavePackageFromWhich()
# $1 PACKAGE
{
	echo "$(${SHELLDIR}/../script/core/check-which.sh "$1")"
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
    if [ "${FLAG}" == "2" ];then
        echo "-fopenmp"
    elif [ "${FLAG}" == "3" ];then
        echo "-fopenmp"
    elif [ "${FLAG}" == "4" ];then
        echo "libgomp1"
    else
        exit 22
    fi
}
elif [ "rpm" == "${KIT_NAME}" ];then 
{
    if [ "${FLAG}" == "2" ];then
        echo "-fopenmp"
    elif [ "${FLAG}" == "3" ];then
        echo "-fopenmp"
    elif [ "${FLAG}" == "4" ];then
        echo "libgomp"
    else
        exit 22
    fi
}
else
{
    exit 1
}
fi 

exit 0
