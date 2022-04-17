/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_VMTX_PROTOCOL_H
#define ABCDK_VMTX_PROTOCOL_H

#include "util/general.h"

__BEGIN_DECLS


/*
 * 回显。
 *
 * REQ: cmd(2)
*/
#define ABCDK_VMTX_COMMAND_ECHO 1000

/*
 * 选举主节点。比较IP地址，地址大的为领导者。
 * 
 * REQ: cmd(2) + round(8)
 *   round: 轮次，从1开始。
 * RSP: cmd(2) + errno(4) + round(8) +  opinion(1)  
 *   opinion: 1 同意，2 不同意。
*/
#define ABCDK_VMTX_COMMAND_VOTE 1



__END_DECLS

#endif //ABCDK_VMTX_PROTOCOL_H
