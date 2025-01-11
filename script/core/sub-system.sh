#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
#

#
SHELLDIR=$(cd `dirname $0`; pwd)

# 
# 1：子系纺启动后，可以通过下面的方法安装或更新软件包。
# #apt update
# #apt install net-tools vim  
#
# 2：子系统启动后，可以编辑下面的文件配置DNS解析。
# #vim /etc/resolv.conf
# #echo "nameserver 8.8.8.8" >> /etc/resolv.conf
#


#挂载.
mnt() 
{
	echo "正在挂载……"
	mount -t proc /proc ${1}/proc
	mount -t sysfs /sys ${1}/sys
	mount -o bind /dev ${1}/dev
	mount -o bind /dev/pts ${1}/dev/pts
	chroot ${1}
	echo "挂载完成。"
}

#卸载.
umnt() 
{
	echo "正在卸载……"
	umount ${1}/proc
	umount ${1}/sys
	umount ${1}/dev/pts
	umount ${1}/dev
	echo "卸载完成。"
}

if [ "$(id -u)" -ne 0 ]; then
{
	echo "权限不足，仅允许root用户执行操作。"
	exit 1
}
elif [ "$1" == "-m" ] && [ -n "$2" ]; then
{
	$(mnt "$2")
	exit $?
}
elif [ "$1" == "-u" ] && [ -n "$2" ]; then
{
	$(umnt "$2")
	exit $?
}
else
{
	echo "至少需要两个参数。"
	echo "第一个参数是操作码。支持挂载(-m)或卸载(-u)两个操作码。"
	echo "第二个参数是rootfs路径。必需是完整路径，否则可能发生意料之外的错误。"
	exit 22
}
fi

