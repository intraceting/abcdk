/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_LZ4_H
#define ABCDK_UTIL_LZ4_H

#include "abcdk/util/general.h"

#ifdef  HAVE_LZ4
#include <lz4.h>
#endif //HAVE_LZ4

__BEGIN_DECLS

#ifdef LZ4_VERSION_NUMBER

/**
 * 解压(safe)。
 * 
 * @param plaintext 明文的指针。
 * @param plaintext_size 明文的长度。
 * @param ciphertext 密文的指针。
 * @param ciphertext_size 密文的长度。
 * 
 * @return > 0 成功(已经解压的密文长度)，<= 0 失败(密文格式错误)。
*/
int abcdk_lz4_dec(void* plaintext, int plaintext_size, const void* ciphertext, int ciphertext_size);

/**
 * 压缩(default)。
 * 
 * @param ciphertext 密文的指针。
 * @param ciphertext_max 密文最大的长度。
 * @param plaintext 明文的指针。
 * @param plaintext_size 明文的长度。
 * 
 * @return > 0 成功(已经压缩的密文长度)，<= 0 失败(密文空间不足)。
*/
int abcdk_lz4_enc(void* ciphertext, int ciphertext_max, const void* plaintext, int plaintext_size);


#endif //LZ4_VERSION_NUMBER

__END_DECLS

#endif //ABCDK_UTIL_LZ4_H