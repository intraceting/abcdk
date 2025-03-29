/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/torch.h"

int abcdk_torch_init_host(uint32_t flags)
{
    return 0;
}

int abcdk_torch_get_runtime_version_host(int *minor)
{
    if (minor)
        *minor = ABCDK_VERSION_MINOR;

    return ABCDK_VERSION_MAJOR;
}

int abcdk_torch_get_device_name_host(char name[256], int id)
{
    if (id >= 1)
        return -1;

#ifdef __x86_64__
    strcpy(name, "CPU x86 64-bit");
#elif defined(__i386__)
    strcpy(name, "CPU x86 32-bit");
#elif defined(__aarch64__)
    strcpy(name, "CPU ARM 64-bit");
#elif defined(__arm__)
    strcpy(name, "CPU ARM 32-bit");
#else
    strcpy(name, "CPU General");
#endif

    return 0;
}