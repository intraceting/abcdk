#!/bin/bash
#
# This file is part of ABCDK.
#  
# MIT License
##

#
CURDIR=$(cd `dirname $0`; pwd)

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
    if [ -f /etc/os-release ];then 
	    grep '^ID=' /etc/os-release |cut -d = -f 2 |sed 's/\"//g' | grep -iE "${1}" |wc -l
    elif [ -f /usr/lib/os-release ];then 
        grep '^ID=' /usr/lib/os-release |cut -d = -f 2 |sed 's/\"//g' | grep -iE "${1}" |wc -l
    else 
        echo "0"
    fi 
}

#
GetSystemVersion()
{
    if [ -f /etc/os-release ];then 
	    grep '^VERSION_ID=' /etc/os-release |cut -d = -f 2 |sed 's/\"//g'
    elif [ -f /usr/lib/os-release ];then 
        grep '^VERSION_ID=' /usr/lib/os-release |cut -d = -f 2 |sed 's/\"//g'
    else 
        echo "0"
    fi 
}

#
CheckPackageKitName()
{
	if [ $(CheckSystemName "Ubuntu|Debian") -ge 1 ];then
		echo "deb"
	elif [ $(CheckSystemName "CentOS|Red Hat|RedHat|RHEL|fedora|Amazon|amzn|Oracle") -ge 1 ];then
		echo "rpm"
	else
		echo ""
	fi
}

#
CheckHavePackageFromKit()
# $1 KIT_NAME
# $1 PACKAGE
{
    #
    STATUS="1"

    #
    KITNAME="$1"
    PACKAGE="$2"

    #
	if [ "deb" == "${KITNAME}" ];then 
        STATUS=$(dpkg -V ${PACKAGE} >> /dev/null 2>&1 ; echo $?)
	elif [ "rpm" == "${KITNAME}" ];then
		STATUS=$(rpm -q ${PACKAGE} >> /dev/null 2>&1 ; echo $?)
    fi

	#
	echo "${STATUS}"
}

#
CheckHavePackageFromWhich()
# $1 KIT_NAME
# $2 PACKAGE
{
    #
    STATUS="1"

    #
    KIT_NAME="$1"
    PACKAGE="$2"

    #
    STATUS=$(which ${PACKAGE} >> /dev/null 2>&1 ; echo $?)

	#
	echo "${STATUS}"
}

#
CheckHavePackage()
# $1 KIT_NAME
# $2 PKG_NAME
# $3 FLAG
{
    #
    KIT_NAME="$1"
    PKG_NAME="$2"
    FLAG="$3"

    #
    SYS_VERID=$(GetSystemVersion)

    #
	if [ "deb" == "${KIT_NAME}" ];then 
	{  
        if [ "${PKG_NAME}" == "pkgconfig" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromWhich ${KIT_NAME} pkg-config)"
            else 
                echo "pkg-config"
            fi
        }
        elif [ "${PKG_NAME}" == "openmp" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libgomp1)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} unixodbc-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libsqlite3-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libssl-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} "libswscale-dev libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev")"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs libswscale libavutil libavcodec libavformat libavdevice libavfilter libavresample libpostproc libswresample)"
            else
                echo "libswscale-dev libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev"
            fi
        }
        elif [ "${PKG_NAME}" == "freeimage" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libfreeimage-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libfuse-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libnm-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} liblz4-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} zlib1g-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libarchive-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libmodbus-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libusb-1.0-0-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libmosquitto-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libhiredis-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libjson-c-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libbluetooth-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libblkid-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libcap-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libfcgi-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libsystemd-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libudev-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libdmtx-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libqrencode-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libzbar-dev)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libmagickwand-dev)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags MagickWand)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs MagickWand)"
            else
                echo "libmagickwand-dev"
            fi
        }
        else
            echo "1"
        fi
    }
	elif [ "rpm" == "${KIT_NAME}" ];then
	{
        if [ "${PKG_NAME}" == "pkgconfig" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromWhich ${KIT_NAME} pkg-config)"
            else
                echo "pkgconfig"
            fi
        }
        elif [ "${PKG_NAME}" == "openmp" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libgomp)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} unixODBC-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags odbc)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs odbc)"
            else
                echo "unixODBC-devel"
            fi
        }
        elif [ "${PKG_NAME}" == "sqlite" ];then
        {
            if [ ${FLAG} -eq 1 ];then
                echo "$(CheckHavePackageFromKit ${KIT_NAME} sqlite-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} openssl-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} ffmpeg-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} freeimage-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} fuse-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} NetworkManager-libnm-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} lz4-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} zlib-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libarchive-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libmodbus-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libusbx-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} mosquitto-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} hiredis-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} json-c-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} bluez-libs-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libblkid-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libcap-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} fcgi-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} systemd-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} systemd-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} libdmtx-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} qrencode-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} zbar-devel)"
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
                echo "$(CheckHavePackageFromKit ${KIT_NAME} ImageMagick-devel)"
            elif [ ${FLAG} -eq 2 ];then
                echo "$(pkg-config --cflags MagickWand)"
            elif [ ${FLAG} -eq 3 ];then
                echo "$(pkg-config --libs MagickWand)"
            else
                echo "ImageMagick-devel"
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


#
MAKE_CONF=${CURDIR}/build/makefile.conf

#
KIT_NAME=$(CheckPackageKitName)

#
SOLUTION_NAME=abcdk

#
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BUILD_PATH=$(realpath "${CURDIR}/build/")


#主版本
VERSION_MAJOR="1"
#副版本
VERSION_MINOR="3"
#发行版本
VERSION_RELEASE="6"

#
BUILD_TYPE="release"

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
    -g  
     生成调试符号。默认：关闭

     自定义编译器，并且定义环境变量。如下：
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
     libudev,dmtx,qrencode,zbar,magickwand

     自定义依赖项，key前缀增加“with-”，并且定义环境变量。如下：
     export DEPEND_FLAGS="-I/tmp/3party/include/"
     export DEPEND_LIBS="-l:3party.so -l:3party.a -l3party -L/tmp/3party/lib/"
EOF
}

#
while getopts "hgV:v:r:i:d:" ARGKEY 
do
    case $ARGKEY in
    h)
        PrintUsage
        exit 22
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
STATUS=$(CheckHavePackageFromWhich ${KIT_NAME} ${CC})
if [ ${STATUS} -ne 0 ];then
{
    echo "${CC} not found."
    exit 22
}
fi

#
STATUS=$(CheckHavePackageFromWhich ${KIT_NAME} ${AR})
if [ ${STATUS} -ne 0 ];then
{
    echo "${AR} not found."
    exit 22
}
fi

#
STATUS=$(CheckHavePackage ${KIT_NAME} pkgconfig 1)
if [ ${STATUS} -ne 0 ];then
{
    echo "$(CheckHavePackage ${KIT_NAME} pkgconfig 0) not found."
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
        }
        else
        {
            CHK=$(CheckHavePackage ${KIT_NAME} ${PACKAGE_KEY} 1)
            if [ ${CHK} -eq 0 ];then
            {
                DEPEND_FLAGS=" -D${PACKAGE_DEF} $(CheckHavePackage ${KIT_NAME} ${PACKAGE_KEY} 2) ${DEPEND_FLAGS}"
                DEPEND_LIBS=" $(CheckHavePackage ${KIT_NAME} ${PACKAGE_KEY} 3) ${DEPEND_LIBS}"
            }
            else
            {
                DEPEND_NOFOUND="$(CheckHavePackage ${KIT_NAME} ${PACKAGE_KEY} 0) ${DEPEND_NOFOUND}"
            }
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

#
if [ "${DEPEND_NOFOUND}" != "" ];then
{
    echo "${DEPEND_NOFOUND} no found."
    exit 22
}
fi 

#
TARGET_PLATFORM=$(${CC} -dumpmachine)

#
VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_RELEASE}

#
mkdir -p ${BUILD_PATH}

#
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
    INSTALL_PREFIX="${INSTALL_PREFIX}/${SOLUTION_NAME}-${VERSION_STR}/"
}
fi

#
DEPEND_FLAGS="${DEPEND_FLAGS} -D_GNU_SOURCE -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64"

#
DEPEND_LIBS="${DEPEND_LIBS} -ldl -pthread -lrt -lc -lm"

#
echo "MAKE_CONF=${MAKE_CONF}"

#
echo "SOLUTION_NAME=${SOLUTION_NAME}"

#
echo "BUILD_TIME=${BUILD_TIME}"
echo "BUILD_PATH=${BUILD_PATH}"

#
echo "TARGET_PLATFORM=${TARGET_PLATFORM}"

#
echo "CC=${CC}"
echo "AR=${AR}"

#
echo "VERSION_MAJOR=${VERSION_MAJOR}"
echo "VERSION_MINOR=${VERSION_MINOR}"
echo "VERSION_RELEASE=${VERSION_RELEASE}"
echo "VERSION_STR=${VERSION_STR}"

#
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "INSTALL_PREFIX=${INSTALL_PREFIX}"
echo "ROOT_PATH?=/"

#
echo "#" > ${MAKE_CONF}
checkReturnCode

#
echo "# A bad c development kit." >> ${MAKE_CONF}
echo "#" >> ${MAKE_CONF}
echo "" >> ${MAKE_CONF}

#
echo "SOLUTION_NAME = ${SOLUTION_NAME}" >> ${MAKE_CONF}

#
echo "BUILD_TIME = ${BUILD_TIME}" >> ${MAKE_CONF}
echo "BUILD_PATH = ${BUILD_PATH}" >> ${MAKE_CONF}

#
echo "TARGET_PLATFORM = ${TARGET_PLATFORM}" >> ${MAKE_CONF}


#
echo "CC = ${CC}" >> ${MAKE_CONF}
echo "AR = ${AR}" >> ${MAKE_CONF}

#
echo "VERSION_MAJOR = ${VERSION_MAJOR}" >> ${MAKE_CONF}
echo "VERSION_MINOR = ${VERSION_MINOR}" >> ${MAKE_CONF}
echo "VERSION_RELEASE = ${VERSION_RELEASE}" >> ${MAKE_CONF}
echo "VERSION_STR = ${VERSION_STR}" >> ${MAKE_CONF}

#
echo "DEPEND_FLAGS = ${DEPEND_FLAGS}" >> ${MAKE_CONF}
echo "DEPEND_LIBS = ${DEPEND_LIBS}" >> ${MAKE_CONF}

#
echo "BUILD_TYPE = ${BUILD_TYPE}" >> ${MAKE_CONF}

#
echo "INSTALL_PREFIX = ${INSTALL_PREFIX}" >> ${MAKE_CONF}

#
echo "ROOT_PATH ?= /" >> ${MAKE_CONF}


