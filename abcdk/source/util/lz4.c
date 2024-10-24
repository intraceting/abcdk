/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk/util/lz4.h"

#ifdef LZ4_VERSION_NUMBER

int abcdk_lz4_dec(void* plaintext, int plaintext_size, const void* ciphertext, int ciphertext_size)
{
    assert(plaintext != NULL && plaintext_size > 0 && ciphertext != NULL);

    return LZ4_decompress_safe(ciphertext,plaintext,ciphertext_size,plaintext_size);
}

int abcdk_lz4_enc(void* ciphertext, int ciphertext_max, const void* plaintext, int plaintext_size)
{
    assert(ciphertext != NULL && ciphertext_max > 0 && plaintext != NULL && plaintext_size > 0 );

    return LZ4_compress_default(plaintext,ciphertext,plaintext_size,ciphertext_max);
}

#endif //LZ4_VERSION_NUMBER