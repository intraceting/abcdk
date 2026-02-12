/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#ifndef ABCDK_QRCODE_QRCODE_H
#define ABCDK_QRCODE_QRCODE_H

#include "abcdk/util/general.h"

#ifdef HAVE_QRENCODE
#include <qrencode.h>
#endif //HAVE_QRENCODE

#ifndef QRENCODE_H

/**
 * Level of error correction.
 */
typedef enum {
	QR_ECLEVEL_L = 0, ///< lowest
	QR_ECLEVEL_M,
	QR_ECLEVEL_Q,
	QR_ECLEVEL_H      ///< highest
} QRecLevel;

#endif //QRENCODE_H

#endif //ABCDK_QRCODE_QRCODE_H