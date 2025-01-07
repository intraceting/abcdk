#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
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
        exit $(CheckHavePackageFromKit "libusb-1.0-0-dev")
    elif [ ${FLAG} -eq 2 ];then
        pkg-config --cflags libusb-1.0 2>/dev/null
    elif [ ${FLAG} -eq 3 ];then
        pkg-config --libs libusb-1.0 2>/dev/null
    elif [ ${FLAG} -eq 4 ];then
        echo "libusb-1.0-0-dev"
    else
        exit 22
    fi
}
elif [ "rpm" == "${KIT_NAME}" ];then 
{
    if [ ${FLAG} -eq 1 ];then
        exit $(CheckHavePackageFromKit "libusbx-devel")
    elif [ ${FLAG} -eq 2 ];then
        pkg-config --cflags libusb-1.0 2>/dev/null
    elif [ ${FLAG} -eq 3 ];then
        pkg-config --libs libusb-1.0 2>/dev/null
    elif [ ${FLAG} -eq 4 ];then
        echo "libusbx-devel"
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
