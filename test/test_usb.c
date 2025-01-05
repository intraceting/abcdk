/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

#ifdef HAVE_LIBUSB
#include <libusb-1.0/libusb.h>
#endif // HAVE_LIBUSB

#define VENDOR_ID 0x0438
#define PRODUCT_ID 0x7900

int abcdk_test_usb(abcdk_option_t *args)
{
#ifdef HAVE_LIBUSB
    libusb_context *ctx = NULL;
    libusb_device_handle *handle = NULL;
    libusb_device **devs;
    ssize_t cnt;
    int rc;

    // 初始化 libusb
    rc = libusb_init(&ctx);
    if (rc < 0) {
        printf("libusb_init failed\n");
        return 1;
    }

    // 获取所有设备
    cnt = libusb_get_device_list(ctx, &devs);
    if (cnt < 0) {
        printf("libusb_get_device_list failed\n");
        libusb_exit(ctx);
        return 1;
    }

    // 假设我们选择第一个设备
    handle = libusb_open_device_with_vid_pid(ctx, VENDOR_ID, PRODUCT_ID);  // 使用实际的 VID 和 PID

    if (handle == NULL) {
        printf("Error opening device\n");
        libusb_free_device_list(devs, 1);
        libusb_exit(ctx);
        return 1;
    }

    // 获取设备描述符
    struct libusb_device_descriptor dev_desc;
    rc = libusb_get_device_descriptor(libusb_get_device(handle), &dev_desc);
    if (rc < 0) {
        printf("Error getting device descriptor\n");
        libusb_close(handle);
        libusb_free_device_list(devs, 1);
        libusb_exit(ctx);
        return 1;
    }

    printf("Device Descriptor:\n");
    printf("  Vendor ID: 0x%04x\n", dev_desc.idVendor);
    printf("  Product ID: 0x%04x\n", dev_desc.idProduct);

    // 获取配置描述符
    struct libusb_config_descriptor *config_desc;
    rc = libusb_get_config_descriptor(libusb_get_device(handle), 0, &config_desc);  // 第0配置
    if (rc < 0) {
        printf("Error getting config descriptor\n");
        libusb_close(handle);
        libusb_free_device_list(devs, 1);
        libusb_exit(ctx);
        return 1;
    }

    // 列出配置中的每个接口
    for (int i = 0; i < config_desc->bNumInterfaces; i++) {
        const struct libusb_interface *interface = &config_desc->interface[i];
        printf("Interface %d:\n", i);
        
        for (int j = 0; j < interface->num_altsetting; j++) {
            const struct libusb_interface_descriptor *interface_desc = &interface->altsetting[j];
            printf("  Interface Number: %d\n", interface_desc->bInterfaceNumber);
            
            // 查找接口中的端点
            for (int k = 0; k < interface_desc->bNumEndpoints; k++) {
                const struct libusb_endpoint_descriptor *endpoint_desc = &interface_desc->endpoint[k];
                
                if (endpoint_desc->bEndpointAddress & LIBUSB_ENDPOINT_IN) {
                    printf("    IN Endpoint: 0x%02x\n", endpoint_desc->bEndpointAddress);
                } else {
                    printf("    OUT Endpoint: 0x%02x\n", endpoint_desc->bEndpointAddress);
                }
            }
        }
    }

    // 清理
    libusb_free_config_descriptor(config_desc);
    libusb_close(handle);
    libusb_free_device_list(devs, 1);
    libusb_exit(ctx);

    return 0;
#endif // HAVE_LIBUSB
}
