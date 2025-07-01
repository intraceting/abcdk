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

    const char *dev_p = abcdk_option_get(args, "--dev", 0, "/dev/gpiochip0");

    int fd = abcdk_open(dev_p, 1, 0, 0);
    assert(fd >= 0);

    int chk = ioctl(fd, GPIO_GET_CHIPINFO_IOCTL, &info);
    assert(chk >= 0);

    abcdk_trace_printf(LOG_DEBUG, "name: %s\nlabel: %s\nlines: %d\n", info.name, info.label, info.lines);

    for (int i = 0; i < info.lines; i++)
    {
        struct gpioline_info line_info;

        line_info.line_offset = i;

        chk = ioctl(fd, GPIO_GET_LINEINFO_IOCTL, &line_info);
        assert(chk >= 0);

        abcdk_trace_printf(LOG_DEBUG, "-------\noffset: %d\nflags: %08X\nname: %s\nconsumer: %s\n", line_info.line_offset, line_info.flags, line_info.name, line_info.consumer);
    }

    abcdk_closep(&fd);
}

float _abcdk_test_gpio_2_dis_measure(int trig_fd, int echo_fd)
{
    struct gpiohandle_data trig_data = {0};
    struct gpiohandle_data echo_data = {0};
    int chk;

    trig_data.values[0] = 0;
    chk = ioctl(trig_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, &trig_data);
    assert(chk >= 0);

    usleep(2);// 2 microseconds pulse

    trig_data.values[0] = 1;
    chk = ioctl(trig_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, &trig_data);
    assert(chk >= 0);

    usleep(10); // 10 microseconds pulse

    trig_data.values[0] = 0;
    chk = ioctl(trig_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, &trig_data);
    assert(chk >= 0);

    uint64_t start_us = 0, end_us = 0;
    uint64_t waited_us = 0;

    while (1)
    {
        chk = ioctl(echo_fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &echo_data);
        assert(chk >= 0);

        if (echo_data.values[0])
        {
            start_us = abcdk_time_systime(6);
            break;
        }

        usleep(2);
        waited_us += 2;

        if (waited_us > 30 * 1000)
            return -0.000001;
    }

    waited_us = 0;
    while (1)
    {
        chk = ioctl(echo_fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &echo_data);
        assert(chk >= 0);

        if (!echo_data.values[0])
        {
            end_us = abcdk_time_systime(6);
            break;
        }

        usleep(2);
        waited_us += 2;

        if (waited_us > 30 * 1000)
            return -0.000001;
    }

    double duration_us = (double)(end_us - start_us);

    double distance_cm = (duration_us * 0.0343) / 2.0;

    return distance_cm;
}

int abcdk_test_gpio_2(abcdk_option_t *args)
{
    struct gpiochip_info info;

    const char *dev_p = abcdk_option_get(args, "--dev", 0, "/dev/gpiochip0");
    int trig_pin = abcdk_option_get_int(args, "--trig-pin", 0, 5);
    int echo_pin = abcdk_option_get_int(args, "--echo-pin", 0, 6);

    int fd = abcdk_open(dev_p, 1, 0, 0);
    assert(fd >= 0);

    struct gpiohandle_request trig_req = {0};
    trig_req.lineoffsets[0] = trig_pin;
    trig_req.lines = 1;
    trig_req.flags = GPIOHANDLE_REQUEST_OUTPUT;
    trig_req.default_values[0] = 0; // Initially LOW
    int chk = ioctl(fd, GPIO_GET_LINEHANDLE_IOCTL, &trig_req);
    assert(chk >= 0);

    struct gpiohandle_request echo_req = {0};
    echo_req.lineoffsets[0] = echo_pin;
    echo_req.lines = 1;
    echo_req.flags = GPIOHANDLE_REQUEST_INPUT;
    chk = ioctl(fd, GPIO_GET_LINEHANDLE_IOCTL, &echo_req);
    assert(chk >= 0);

    fprintf(stdout,"\n");

    for (int i = 0; i < 999999999; i++)
    {
        float dist = _abcdk_test_gpio_2_dis_measure(trig_req.fd, echo_req.fd);

        abcdk_trace_printf(LOG_DEBUG, "dist: %.2fm", dist / 100);

        //   fprintf(stdout, "\rdist: %.2fm", dist / 100);

        usleep(500 * 1000);
    }

    abcdk_closep(&trig_req.fd);
    abcdk_closep(&echo_req.fd);
    abcdk_closep(&fd);
}

int abcdk_test_gpio(abcdk_option_t *args)
{
    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    if (cmd == 1)
        return abcdk_test_gpio_1(args);
    if (cmd == 2)
        return abcdk_test_gpio_2(args);
}

#endif // HAVE_GPIO_H