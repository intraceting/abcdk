/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "service.h"

void abcdk_vmc_start(abcdk_vmc_t *ctx)
{
    abcdk_comm_start(1);

    
}

void abcdk_vmc_stop(abcdk_vmc_t *ctx)
{
    abcdk_comm_stop();
}