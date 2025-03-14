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
        exit $(CheckHavePackageFromKit libopencv-dev)
    elif [ ${FLAG} -eq 2 ];then
    {
        CFLAG="-I$(FindIncPath opencv2/opencv.hpp)"
        if [ $? == 0 ];then
        {
            echo "${CFLAG}"
        }
        else 
        {
            CFLAG="-I$(FindIncPath opencv4/opencv2/opencv.hpp)"
            checkReturnCode

            echo "${CFLAG}/opencv4"
        }
        fi
        
    }
    elif [ ${FLAG} -eq 3 ];then
    {
        LDFLAG="-L$(FindLibPath libopencv_xfeatures2d.so)"
        if [ $? -eq 0 ];then
        {
            echo "-lopencv_calib3d -lopencv_flann -lopencv_highgui -lopencv_freetype -lopencv_dnn -lopencv_photo -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_stitching -lopencv_features2d -lopencv_xfeatures2d   ${LDFLAG}"
        }
        else 
        {
            LDFLAG="-L$(FindLibPath libopencv_core.so)"
            checkReturnCode

            echo "-lopencv_calib3d -lopencv_flann -lopencv_highgui -lopencv_freetype -lopencv_dnn -lopencv_photo -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_stitching -lopencv_features2d ${LDFLAG}"
        }
        fi
    }
    elif [ ${FLAG} -eq 4 ];then
        echo "libopencv-dev"
    else
        exit 22
    fi
}
elif [ "rpm" == "${KIT_NAME}" ];then 
{
    if [ ${FLAG} -eq 1 ];then
        exit $(CheckHavePackageFromKit opencv-devel)
    elif [ ${FLAG} -eq 2 ];then
    {
        CFLAG="-I$(FindIncPath opencv2/opencv.hpp)"
        if [ $? != 0 ];then
        {
            echo "${CFLAG}"
        }
        else 
        {
            CFLAG="-I$(FindIncPath opencv4/opencv2/opencv.hpp)"
            checkReturnCode

            echo "${CFLAG}/opencv4"
        }
        fi
    }
    elif [ ${FLAG} -eq 3 ];then
    {
        LDFLAG="-L$(FindLibPath libopencv_xfeatures2d.so)"
        if [ $? -eq 0 ];then
        {
            echo "-lopencv_calib3d -lopencv_highgui -lopencv_freetype -lopencv_dnn -lopencv_photo -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_stitching -lopencv_features2d -lopencv_xfeatures2d   ${LDFLAG}"
        }
        else 
        {
            LDFLAG="-L$(FindLibPath libopencv_core.so)"
            checkReturnCode

            echo "-lopencv_calib3d -lopencv_highgui -lopencv_freetype -lopencv_dnn -lopencv_photo -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_stitching -lopencv_features2d ${LDFLAG}"
        }
        fi
    }
    elif [ ${FLAG} -eq 4 ];then
        echo "opencv-devel"
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
