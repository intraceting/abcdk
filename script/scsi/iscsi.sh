#!/bin/bash
#
# This file is part of ABCDK.
#  
# Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
# 
#


# 以下代码来自于网友，出处无法考证。

ipAddr=$2
DRIVERDIR=/lib/modules/$(uname -r)/kernel/drivers/scsi
OP=$1
if  [ -z $ipAddr ]
then
  echo "ip is null"
        echo $"Usage:$0 {iscsi start|stop IPADDRESS}"
        exit 1
fi

libiscsi=libiscsi/$(lsmod | grep '^libiscsi' | awk '{print $1}')
iscsi_tcp=iscsi_tcp/$(lsmod | grep '^iscsi_tcp' | awk '{print $1}')
scsi_transport_iscsi=scsi_transport_iscsi/$(lsmod | grep '^scsi_transport_iscsi '| awk '{print $1}')
for NEEDMOD in  $scsi_transport_iscsi $libiscsi $iscsi_tcbiscsi $iscsi_tcpp
do
  suffix=${NEEDMOD#*/}
  prefix=${NEEDMOD%/*}
 
  if [  -z $suffix ]
  then
    if [ -f $DRIVERDIR/$prefix.ko ]
    then
       insmod $DRIVERDIR/$prefix.ko
    else
       echo "need mod ${$prefix}.ko"
       exit 1
    fi
  fi
done
start()
{
  echo "start to connect target in"
  iscsid
  cmdQuery=$(iscsiadm -m discovery -t sendtargets -p ${ipAddr}:3260 | sed 's/.* //g')
  targetName=$cmdQuery
  echo ${targetName}
  iscsiadm -m node -T ${targetName} -p ${ipAddr}:3260 --login
}
stop()
{
  cmdQuery=$(iscsiadm -m discovery -t sendtargets -p ${ipAddr}:3260 | sed 's/.* //g')
 targetName=$cmdQuery
 iscsiadm -m node -T ${targetName} -p ${ipAddr}:3260 --logout
}
case $OP in
start)
      start
      ;;
stop)
     stop
     ;;
*)
    echo $"Usage:$0 {iscsi start|stop IPADDRESS}"
    exit 2
esac 
