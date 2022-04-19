/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_VMTX_ENTRY_H
#define ABCDK_VMTX_ENTRY_H

#include "util/general.h"
#include "util/option.h"
#include "util/getargs.h"
#include "shell/proc.h"
#include "util/log.h"

__BEGIN_DECLS

int abcdk_vmtx_server(abcdk_tree_t *args);
int abcdk_vmtx_client(abcdk_tree_t *args);

__END_DECLS

#endif //ABCDK_VMTX_ENTRY_H
