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
    echo "$(${SHELLDIR}/myscript/linux/core/check-os-id.sh "$1")"
}

#
GetSystemVersion()
{
    echo "$(${SHELLDIR}/myscript/linux/core/get-os-ver.sh)"
}

#
CheckPackageKitName()
{
	echo "$(${SHELLDIR}/myscript/linux/core/get-kit-name.sh)"
}

#
CheckHavePackageFromKit()
# $1 PACKAGE
{
    echo "$(${SHELLDIR}/myscript/linux/core/check-package.sh "$1")"
}

#
CheckHavePackageFromWhich()
# $1 PACKAGE
{
	echo "$(${SHELLDIR}/myscript/linux/core/check-which.sh "$1")"
}

#
CheckHavePackage()
# $1 PKG_NAME
# $2 FLAG
{
    #
    SYS_VERID=$(GetSystemVersion)
    KIT_NAME=$(CheckPackageKitName)
    PKG_NAME="$1"
    FLAG="$2"

    #
	if [ "deb" == "${KIT_NAME}" ];then 
	{  
        if [ "${PKG_NAME}" == "binutils" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit binutils)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo ""
            else 
                echo "binutils"
            fi
        }
        elif [ "${PKG_NAME}" == "pkgconfig" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromWhich pkg-config)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo ""
            else 
                echo "pkg-config"
            fi
        }
        elif [ "${PKG_NAME}" == "openmp" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libgomp1)"
            elif [ ${FLAG} -eq 2 ];then
                echo "-fopenmp"
            elif [ ${FLAG} -eq 3 ];then
                echo "-fopenmp"
            else
                echo "libgomp1"
            fi
        }
        elif [ "${PKG_NAME}" == "unixodbc" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit unixodbc-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lodbc"
            else
                echo "unixodbc-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "sqlite" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libsqlite3-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags sqlite3)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs sqlite3)"
            else
                echo "libsqlite3-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "openssl" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libssl-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags openssl)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs openssl)"
            else
                echo "libssl-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "ffmpeg" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit "libswscale-dev libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libavresample-dev libpostproc-dev libswresample-dev")"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            else
                echo "libswscale-dev libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libavresample-dev libpostproc-dev libswresample-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "freeimage" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libfreeimage-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lfreeimage"
            else
                echo "libfreeimage-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "fuse" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libfuse-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags fuse)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs fuse)"
            else
                echo "libfuse-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "libnm" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libnm-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libnm)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libnm)"
            else
                echo "libnm-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "lz4" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit liblz4-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags liblz4)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs liblz4)"
            else
                echo "liblz4-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "zlib" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit zlib1g-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags zlib)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs zlib)"
            else
                echo "zlib1g-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "archive" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libarchive-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libarchive)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libarchive)"
            else
                echo "libarchive-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "modbus" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libmodbus-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libmodbus)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libmodbus)"
            else
                echo "libmodbus-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "libusb" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libusb-1.0-0-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libusb-1.0)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libusb-1.0)"
            else
                echo "libusb-1.0-0-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "mqtt" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libmosquitto-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lmosquitto"
            else
                echo "libmosquitto-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "redis" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libhiredis-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags hiredis)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs hiredis)"
            else
                echo "libhiredis-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "json-c" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libjson-c-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags json-c)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs json-c)"
            else
                echo "libjson-c-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "bluez" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libbluetooth-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags bluez)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs bluez)"
            else
                echo "libbluetooth-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "blkid" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libblkid-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags blkid)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs blkid)"
            else
                echo "libblkid-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "libcap" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libcap-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libcap)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libcap)"
            else
                echo "libcap-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "fastcgi" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libfcgi-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lfcgi"
            else
                echo "libfcgi-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "systemd" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libsystemd-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libsystemd)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libsystemd)"
            else
                echo "libsystemd-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "libudev" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libudev-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libudev)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libudev)"
            else
                echo "libudev-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "dmtx" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libdmtx-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libdmtx)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libdmtx)"
            else
                echo "libdmtx-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "qrencode" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libqrencode-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libqrencode)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libqrencode)"
            else
                echo "libqrencode-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "zbar" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libzbar-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags zbar)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs zbar)"
            else
                echo "libzbar-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "magickwand" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libmagickwand-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags MagickWand)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs MagickWand)"
            else
                echo "libmagickwand-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "kafka" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit librdkafka-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags rdkafka)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs rdkafka)"
            else
                echo "librdkafka-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "uuid" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit uuid-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags uuid)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs uuid)"
            else
                echo "uuid-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "openblas" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libopenblas-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags lapack-openblas blas-openblas)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs lapack-openblas blas-openblas)"
            else
                echo "libopenblas-dev"
            fi
        }
        else
        {
            if [ ${FLAG} -eq 1 ];then 
                echo "1"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo ""
            else
                echo "${PKG_NAME}"
            fi
        }
        fi
    }
	elif [ "rpm" == "${KIT_NAME}" ];then
	{
        if [ "${PKG_NAME}" == "binutils" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit binutils)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo ""
            else 
                echo "binutils"
            fi
        }
        elif [ "${PKG_NAME}" == "pkgconfig" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromWhich pkg-config)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo ""
            else
                if [ ${SYS_VERID} -eq 7 ];then
                    echo "pkgconfig"
                elif [ ${SYS_VERID} -eq 8 ];then
                    echo "pkgconf-pkg-config"
                else 
                    echo ""
                fi
            fi
        }
        elif [ "${PKG_NAME}" == "openmp" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libgomp)"
            elif [ ${FLAG} -eq 2 ];then
                echo "-fopenmp"
            elif [ ${FLAG} -eq 3 ];then
                echo "-fopenmp"
            else
                echo "libgomp"
            fi
        }
        elif [ "${PKG_NAME}" == "unixodbc" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit unixODBC-devel)"
            elif [ ${FLAG} -eq 2 ];then
                if [ ${SYS_VERID} -eq 7 ];then
                    echo "-DHAVE_UNISTD_H -DHAVE_PWD_H -DHAVE_SYS_TYPES_H -DHAVE_LONG_LONG -DSIZEOF_LONG_INT=8"
                elif [ ${SYS_VERID} -eq 8 ];then
                    echo "$(pkg-config --cflags odbc)"
                else 
                    echo ""
                fi
            elif [ ${FLAG} -eq 3 ];then
                if [ ${SYS_VERID} -eq 7 ];then
                    echo "-lodbc"
                elif [ ${SYS_VERID} -eq 8 ];then
                    echo "$(pkg-config --libs odbc)"
                else 
                    echo ""
                fi
            else 
                echo "unixODBC-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "sqlite" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit sqlite-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags sqlite3)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs sqlite3)"
            else
                echo "sqlite-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "openssl" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit openssl-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags openssl)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs openssl)"
            else
                echo "openssl-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "ffmpeg" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ffmpeg-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            else 
                echo "ffmpeg-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "freeimage" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit freeimage-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lfreeimage"
            else
                echo "freeimage-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "fuse" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit fuse-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags fuse)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs fuse)"
            else
                echo "fuse-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "libnm" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit NetworkManager-libnm-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libnm)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libnm)"
            else
                echo "NetworkManager-libnm-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "lz4" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit lz4-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags liblz4)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs liblz4)"
            else
                echo "lz4-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "zlib" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit zlib-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags zlib)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs zlib)"
            else
                echo "zlib-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "archive" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libarchive-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libarchive)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libarchive)"
            else
                echo "libarchive-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "modbus" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libmodbus-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libmodbus)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libmodbus)"
            else
                echo "libmodbus-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "libusb" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libusbx-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libusb-1.0)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libusb-1.0)"
            else
                echo "libusbx-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "mqtt" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit mosquitto-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libmosquitto)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libmosquitto)"
            else
                echo "mosquitto-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "redis" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit hiredis-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags hiredis)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs hiredis)"
            else
                echo "hiredis-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "json-c" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit json-c-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags json-c)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs json-c)"
            else
                echo "json-c-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "bluez" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit bluez-libs-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags bluez)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs bluez)"
            else
                echo "bluez-libs-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "blkid" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libblkid-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags blkid)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs blkid)"
            else
                echo "libblkid-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "libcap" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libcap-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libcap)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libcap)"
            else
                echo "libcap-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "fastcgi" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit fcgi-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo "-lfcgi"
            else
                echo "fcgi-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "systemd" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit systemd-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libsystemd)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libsystemd)"
            else
                echo "systemd-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "libudev" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit systemd-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libudev)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libudev)"
            else
                echo "systemd-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "dmtx" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libdmtx-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libdmtx)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libdmtx)"
            else
                echo "libdmtx-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "qrencode" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit qrencode-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libqrencode)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libqrencode)"
            else
                echo "qrencode-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "zbar" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit zbar-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags zbar)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs zbar)"
            else
                echo "zbar-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "magickwand" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ImageMagick-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags MagickWand)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs MagickWand)"
            else
                echo "ImageMagick-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "kafka" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit librdkafka-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags rdkafka)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs rdkafka)"
            else
                echo "librdkafka-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "uuid" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit libuuid-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags uuid)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs uuid)"
            else
                echo "libuuid-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "openblas" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit openblas-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "-I/usr/include/openblas"
            elif [ ${FLAG} -eq 3 ];then
                echo "-lopenblas"
            else
                echo "openblas-devel"
            fi
        }
        else
        {
            if [ ${FLAG} -eq 1 ];then 
                echo "1"
            elif [ ${FLAG} -eq 2 ];then
                echo ""
            elif [ ${FLAG} -eq 3 ];then
                echo ""
            else 
                echo "${PKG_NAME}"
            fi
        }
        fi
    }
    else 
        echo "1"
    fi
}

#
CheckKeyword()
# $1 keywords
# $2 word
{
	NUM=$(echo "$1" |grep -wi "$2" | wc -l)
    echo ${NUM}
}

#修改执行权限，不然用不了脚本。
chmod 0755 ${SHELLDIR}/myscript/linux/* -R 

#
KIT_NAME=$(CheckPackageKitName)

#
SOLUTION_NAME="abcdk"

#
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BUILD_PATH=$(realpath "${SHELLDIR}/build/")

#
MAKE_CONF=${BUILD_PATH}/makefile.conf

#
PKG_PC=${BUILD_PATH}/pkg_conf.pc

#
RPM_RT_SPEC=${BUILD_PATH}/rpm_rt.spec
RPM_DEV_SPEC=${BUILD_PATH}/rpm_devel.spec

#
DEB_RT_CTL=${BUILD_PATH}/deb_rt.ctl
DEB_DEV_CTL=${BUILD_PATH}/deb_devel.ctl

#主版本
VERSION_MAJOR="1"
#副版本
VERSION_MINOR="4"
#发行版本
VERSION_RELEASE="1"

#
BUILD_TYPE="release"
BUILD_OPTIMIZE="No"

#
INSTALL_PREFIX="/usr/local/"

#
DEPEND_FUNC="Nothing"
DEPEND_NOFOUND=""
DEPEND_REQUIRES=""

#
PrintUsage()
{
cat << EOF
usage: [ OPTIONS ]

    -h 
     打印帮助信息。

    -O
     编译优化。默认：关闭。

    -g  
     生成调试符号。默认：关闭

     自定义编译器（定义环境变量）。如下：
     export CC=gcc
     export AR=ar

    -V < number > 
     主版本。默认：${VERSION_MAJOR}

    -v < number > 
     副版本。默认：${VERSION_MINOR}

    -r < number > 
     发行版本。默认：${VERSION_RELEASE}

    -i < path > 
     安装路径。默认：${INSTALL_PREFIX}

    -d < key,key,... > 
     依赖项目，以英文“,”为分割符。支持以下关键字：
     openmp,unixodbc,sqlite,openssl,ffmpeg,
     freeimage,fuse,libnm,lz4,zlib,
     archive,modbus,libusb,mqtt,redis,json-c,
     bluez,blkid,libcap,fastcgi,systemd,
     libudev,dmtx,qrencode,zbar,magickwand,
     kafka,uuid

     自定义依赖项（定义环境变量），并且key前缀增加“with-”。如下：
     export DEPEND_FLAGS="-I/tmp/3party/include/"
     export DEPEND_LIBS="-l:3party.so -l:3party.a -l3party -L/tmp/3party/lib/"
EOF
}

#
while getopts "hOgV:v:r:i:d:" ARGKEY 
do
    case $ARGKEY in
    h)
        PrintUsage
        exit 22
    ;;
    O)
        BUILD_OPTIMIZE="yes"
    ;;
    g)
        BUILD_TYPE="debug"
    ;;
    V)
        VERSION_MAJOR="${OPTARG}"
    ;;
    v)
        VERSION_MINOR="${OPTARG}"
    ;;
    r)
        VERSION_RELEASE="${OPTARG}"
    ;;
    i)
        INSTALL_PREFIX=$(realpath "${OPTARG}")
    ;;
    d)
        DEPEND_FUNC="${OPTARG}"
    ;;
    esac
done

# 设置编译器。
if [ "${CC}" == "" ];then
    CC=gcc
fi
if [ "${AR}" == "" ];then
    AR=ar
fi

#
TARGET_PLATFORM=$(${CC} -dumpmachine)

#
STATUS=$(CheckHavePackageFromWhich ${CC})
if [ ${STATUS} -ne 0 ];then
{
    echo "${CC} not found."
    exit 22
}
fi

#
STATUS=$(CheckHavePackageFromWhich ${AR})
if [ ${STATUS} -ne 0 ];then
{
    echo "${AR} not found."
    exit 22
}
fi

#
STATUS=$(CheckHavePackage pkgconfig 1)
if [ ${STATUS} -ne 0 ];then
{
    echo "$(CheckHavePackage pkgconfig 0) not found."
    exit 22
}
fi

#
DependPackageCheck()
# 1 key
# 2 def
{
    PACKAGE_KEY=$1
    PACKAGE_DEF=$2
    #
    if [ $(CheckKeyword ${DEPEND_FUNC} ${PACKAGE_KEY}) -eq 1 ];then
    {
        if [ $(CheckKeyword ${DEPEND_FUNC} with-${PACKAGE_KEY}) -eq 1 ];then
        {
            DEPEND_FLAGS=" -D${PACKAGE_DEF} ${DEPEND_FLAGS}"

            echo -e "With ${PACKAGE_KEY}"
        }
        else
        {
            CHK=$(CheckHavePackage ${PACKAGE_KEY} 1)
            if [ ${CHK} -eq 0 ];then
            {
                DEPEND_FLAGS="-D${PACKAGE_DEF} $(CheckHavePackage ${PACKAGE_KEY} 2) ${DEPEND_FLAGS}"
                DEPEND_LIBS="$(CheckHavePackage ${PACKAGE_KEY} 3) ${DEPEND_LIBS}"
            }
            else
            {
                DEPEND_NOFOUND="$(CheckHavePackage ${PACKAGE_KEY} 4) ${DEPEND_NOFOUND}"
            }
            fi

            echo -n "Check ${PACKAGE_KEY}"
            if [ ${CHK} -eq 0 ];then
                echo -e "\x1b[32m Ok \x1b[0m"
            else 
                echo -e "\x1b[31m Failed \x1b[0m"
            fi
        }
        fi
    }
    fi

#    echo ${DEPEND_FLAGS} 
#    echo ${DEPEND_LIBS}
}

#
DependPackageCheck openmp HAVE_OPENMP
DependPackageCheck unixodbc HAVE_UNIXODBC
DependPackageCheck sqlite HAVE_SQLITE
DependPackageCheck openssl HAVE_OPENSSL
DependPackageCheck ffmpeg HAVE_FFMPEG
DependPackageCheck freeimage HAVE_FREEIMAGE
DependPackageCheck fuse HAVE_FUSE
DependPackageCheck libnm HAVE_LIBNM
DependPackageCheck lz4 HAVE_LZ4
DependPackageCheck zlib HAVE_ZLIB
DependPackageCheck archive HAVE_ARCHIVE
DependPackageCheck modbus HAVE_MODBUS
DependPackageCheck libusb HAVE_LIBUSB
DependPackageCheck mqtt HAVE_MQTT
DependPackageCheck redis HAVE_REDIS
DependPackageCheck json-c HAVE_JSON_C
DependPackageCheck bluez HAVE_BLUEZ
DependPackageCheck blkid HAVE_BLKID
DependPackageCheck libcap HAVE_LIBCAP
DependPackageCheck fastcgi HAVE_FASTCGI
DependPackageCheck samba HAVE_SAMBA
DependPackageCheck systemd HAVE_SYSTEMD
DependPackageCheck libudev HAVE_LIBUDEV
DependPackageCheck dmtx HAVE_LIBDMTX
DependPackageCheck qrencode HAVE_QRENCODE
DependPackageCheck zbar HAVE_ZBAR
DependPackageCheck magickwand HAVE_MAGICKWAND
DependPackageCheck kafka HAVE_KAFKA
DependPackageCheck uuid HAVE_UUID

#
if [ "${DEPEND_NOFOUND}" != "" ];then
{
    echo -e "\x1b[33m${DEPEND_NOFOUND}\x1b[31m Not Found \x1b[0m"
    exit 22
}
fi 

#
mkdir -p ${BUILD_PATH}
if [ ! -d ${BUILD_PATH} ];then
{
    echo "'${BUILD_PATH}' must be an existing directory."
    exit 22
}
fi

#
if [ ! -d ${INSTALL_PREFIX} ];then
{
    echo "'${INSTALL_PREFIX}' must be an existing directory."
    exit 22
}
else
{
    INSTALL_PREFIX="${INSTALL_PREFIX}/${SOLUTION_NAME}/"
}
fi

#
INSTALL_PREFIX=$(realpath "${INSTALL_PREFIX}")

#
DEPEND_FLAGS="${DEPEND_FLAGS} -D_GNU_SOURCE -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64"

#
DEPEND_LIBS="${DEPEND_LIBS} -ldl -pthread -lrt -lc -lm"

#
cat >${MAKE_CONF} <<EOF
#
KIT_NAME = ${KIT_NAME}
#
SOLUTION_NAME = ${SOLUTION_NAME}
#
BUILD_TIME = ${BUILD_TIME}
BUILD_PATH = ${BUILD_PATH}
#
TARGET_PLATFORM = ${TARGET_PLATFORM}
#
CC = ${CC}
AR = ${AR}
#
VERSION_MAJOR = ${VERSION_MAJOR}
VERSION_MINOR = ${VERSION_MINOR}
VERSION_RELEASE = ${VERSION_RELEASE}
VERSION_STR = ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
#
DEPEND_FLAGS = ${DEPEND_FLAGS}
DEPEND_LIBS = ${DEPEND_LIBS}
#
BUILD_TYPE = ${BUILD_TYPE}
BUILD_OPTIMIZE = ${BUILD_OPTIMIZE}
#
INSTALL_PREFIX = ${INSTALL_PREFIX}
#
ROOT_PATH ?= /

EOF
checkReturnCode

#
cat >${PKG_PC} <<EOF
prefix=${INSTALL_PREFIX}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: ${SOLUTION_NAME}
Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
Description: The ${SOLUTION_NAME} Libraries
Requires:
Libs: -L\${libdir} -labcdk
Cflags: -I\${includedir}
EOF
checkReturnCode


#
if [ "${KIT_NAME}" == "rpm" ];then
{

#
cat >>${MAKE_CONF} <<EOF
#
PKG_PC = ${PKG_PC}
#
RPM_RT_SPEC = ${RPM_RT_SPEC}
RPM_DEV_SPEC = ${RPM_DEV_SPEC}
EOF
checkReturnCode

#
cat >${RPM_RT_SPEC} <<EOF
Summary: A Better C language Development Kit (a.k.a ABCDK).
Vendor: https://github.com/intraceting/abcdk
Name: ${SOLUTION_NAME}
Version: ${VERSION_MAJOR}.${VERSION_MINOR}
Release: ${VERSION_RELEASE}
Group: Applications/System
License: MIT
AutoReqProv: yes

%description
The C language and C-interface style secondary development kit, 
only supports gnu/linux compatible platforms.
.
This package contains the development files (tools,libraries)
%files
${INSTALL_PREFIX}

%post
#!/bin/sh
echo "export PATH=\\\$PATH:${INSTALL_PREFIX}/bin" > /etc/profile.d/${SOLUTION_NAME}.sh
echo "export LD_LIBRARY_PATH=\\\$LD_LIBRARY_PATH:${INSTALL_PREFIX}/lib" >> /etc/profile.d/${SOLUTION_NAME}.sh
exit 0

%postun
#!/bin/sh
rm -f /etc/profile.d/${SOLUTION_NAME}.sh
exit 0
EOF
checkReturnCode

#
cat >${RPM_DEV_SPEC} <<EOF
Summary: A Better C language Development Kit (a.k.a ABCDK).
Vendor: https://github.com/intraceting/abcdk
Name: ${SOLUTION_NAME}-devel
Version: ${VERSION_MAJOR}.${VERSION_MINOR}
Release: ${VERSION_RELEASE}
Group: Applications/System
License: MIT
Requires: ${SOLUTION_NAME} = ${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_RELEASE}
AutoReqProv: yes

%description
The C language and C-interface style secondary development kit, 
only supports gnu/linux compatible platforms.
.
This package contains the development files (headers, static libraries)

%files
${INSTALL_PREFIX}

%post
#!/bin/sh
echo "export PKG_CONFIG_PATH=\\\$PKG_CONFIG_PATH:${INSTALL_PREFIX}/pkgconfig" >/etc/profile.d/${SOLUTION_NAME}-devel.sh
exit 0

%postun
#!/bin/sh
rm -f /etc/profile.d/${SOLUTION_NAME}-devel.sh
exit 0
EOF
checkReturnCode

}
elif [ "${KIT_NAME}" == "deb" ];then
{

#
mkdir -p ${DEB_RT_CTL}
mkdir -p ${DEB_DEV_CTL}

#
rm -rf ${DEB_RT_CTL}/*
rm -rf ${DEB_DEV_CTL}/*

#
cat >>${MAKE_CONF} <<EOF
#
PKG_PC = ${PKG_PC}
#
DEB_RT_CTL = ${DEB_RT_CTL}
DEB_DEV_CTL = ${DEB_DEV_CTL}
#
DEB_TOOL_ROOT = ${SHELLDIR}/myscript/linux/dpkg
EOF
checkReturnCode

#
cat >${DEB_RT_CTL}/control <<EOF
Source: ${SOLUTION_NAME}
Package: ${SOLUTION_NAME}
Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
Section: Applications/System
Priority: optional
Architecture: amd64
Maintainer: https://github.com/intraceting/abcdk
Pre-Depends: \${shlibs:Depends}
Description: The C language and C-interface style secondary development kit,
 only supports gnu/linux compatible platforms.
 .
 This package contains the development files (tools, libraries)
EOF
checkReturnCode

#
cat >${DEB_RT_CTL}/postinst <<EOF
#!/bin/sh
echo "export PATH=\\\$PATH:${INSTALL_PREFIX}/bin" > /etc/profile.d/${SOLUTION_NAME}.sh
echo "export LD_LIBRARY_PATH=\\\$LD_LIBRARY_PATH:${INSTALL_PREFIX}/lib" >> /etc/profile.d/${SOLUTION_NAME}.sh
exit 0
EOF
checkReturnCode

#
cat >${DEB_RT_CTL}/postrm <<EOF
#!/bin/sh
rm -f /etc/profile.d/${SOLUTION_NAME}.sh
exit 0
EOF
checkReturnCode

#
cat >${DEB_DEV_CTL}/control <<EOF
Source: ${SOLUTION_NAME}
Package: ${SOLUTION_NAME}-devel
Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}
Section: Applications/System
Priority: optional
Architecture: amd64
Maintainer: https://github.com/intraceting/abcdk
Pre-Depends: ${SOLUTION_NAME} (= ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE})
Description: The C language and C-interface style secondary development kit, 
 only supports gnu/linux compatible platforms.
 .
 This package contains the development files (headers, static libraries)
EOF
checkReturnCode

#
cat >${DEB_DEV_CTL}/postinst <<EOF
#!/bin/sh
echo "export PKG_CONFIG_PATH=\\\$PKG_CONFIG_PATH:${INSTALL_PREFIX}/pkgconfig" >/etc/profile.d/${SOLUTION_NAME}-devel.sh
exit 0
EOF
checkReturnCode

#
cat >${DEB_DEV_CTL}/postrm <<EOF
#!/bin/sh
rm -f /etc/profile.d/${SOLUTION_NAME}-devel.sh
exit 0
EOF
checkReturnCode

#
chmod 755 ${DEB_RT_CTL}/postinst
chmod 755 ${DEB_RT_CTL}/postrm
chmod 755 ${DEB_DEV_CTL}/postinst
chmod 755 ${DEB_DEV_CTL}/postrm

}
fi
