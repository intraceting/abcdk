/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_VMTX_DEVICE_H
#define ABCDK_VMTX_DEVICE_H

#include "util/thread.h"
#include "shell/scsi.h"

void abcdk_vmtx_dev_watch(abcdk_tree_t **snapshot, abcdk_tree_t **add, abcdk_tree_t **del);

#endif //ABCDK_VMTX_DEVICE_H