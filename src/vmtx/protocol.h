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
* REQ: cmd(2) + vote(1)
*   vote: 1 选我， 2 选你。
* REQ: cmd(2) + errno(4) + assert(1)
*   assert: 1 同意，2 不同意。
*/
#define ABCDK_VMTX_COMMAND_ELECT_LEADER 1000

__END_DECLS

#endif //ABCDK_VMTX_PROTOCOL_H
