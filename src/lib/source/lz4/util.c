/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#include "abcdk/lz4/util.h"

int abcdk_lz4_dec(void* plaintext, int plaintext_size, const void* ciphertext, int ciphertext_size)
{
#ifndef HAVE_LZ4
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含LZ4工具。"));
    return -1;
#else //#ifdef HAVE_LZ4
    assert(plaintext != NULL && plaintext_size > 0 && ciphertext != NULL);

    return LZ4_decompress_safe(ciphertext,plaintext,ciphertext_size,plaintext_size);
#endif //#ifdef HAVE_LZ4
}

int abcdk_lz4_enc(void* ciphertext, int ciphertext_max, const void* plaintext, int plaintext_size)
{
#ifndef HAVE_LZ4
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含LZ4工具。"));
    return -1;
#else //#ifdef HAVE_LZ4
    assert(ciphertext != NULL && ciphertext_max > 0 && plaintext != NULL && plaintext_size > 0 );

    return LZ4_compress_default(plaintext,ciphertext,plaintext_size,ciphertext_max);
#endif //#ifdef HAVE_LZ4
}


