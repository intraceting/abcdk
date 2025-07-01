/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

#ifdef HAVE_GPIO_H

int abcdk_test_gpio_1(abcdk_option_t *args)
{
    struct gpiochip_info info;

    int fd = abcdk_open("/dev/gpiochip0", 1, 0, 0);
    assert(fd >= 0);

    int chk = ioctl(fd, GPIO_GET_CHIPINFO_IOCTL, &info);
    assert(chk >= 0);

    abcdk_trace_printf(LOG_DEBUG,"name: %s\nlabel: %s\nlines: %d\n",info.name,info.label,info.lines);


    abcdk_closep(&fd);
}

int abcdk_test_gpio(abcdk_option_t *args)
{
    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    if (cmd == 1)
        return abcdk_test_gpio_1(args);
}

#endif // HAVE_GPIO_H