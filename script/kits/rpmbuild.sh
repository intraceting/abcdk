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
    ${SHELLDIR}/../core/check-os-id.sh "$1"
}

#
GetSystemVersion()
{
    ${SHELLDIR}/../core/get-os-ver.sh
}

#
CheckPackageKitName()
{
	${SHELLDIR}/../core/get-kit-name.sh
}

#
CheckHavePackageFromKit()
# $1 PACKAGE
{
    ${SHELLDIR}/../core/check-package.sh "$1"
}

#
CheckHavePackageFromWhich()
# $1 PACKAGE
{
	${SHELLDIR}/../core/check-which.sh "$1"
}

#
FindIncPath()
# $1 HDNAME
{
	${SHELLDIR}/../core/find-inc-path.sh "$1"
}

#
FindLibPath()
# $1 SONAME
{
	${SHELLDIR}/../core/find-lib-path.sh "$1"
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
elif [ "rpm" == "${KIT_NAME}" ];then 
{
    if [ ${FLAG} -eq 1 ];then
        exit $(CheckHavePackageFromKit rpm-build)
    elif [ ${FLAG} -eq 2 ];then
        echo ""
    elif [ ${FLAG} -eq 3 ];then
        echo ""
    elif [ ${FLAG} -eq 4 ];then
        echo "rpm-build"
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
