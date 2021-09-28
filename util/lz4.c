/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk-util/lz4.h"

#ifdef LZ4_VERSION_NUMBER

int abcdk_lz4_dec_fast(void *plaintext, int plaintext_size, const void *ciphertext)
{
    assert(plaintext != NULL && plaintext_size > 0 && ciphertext != NULL);

    return LZ4_decompress_fast(ciphertext,plaintext,plaintext_size);
}

#endif //LZ4_VERSION_NUMBER