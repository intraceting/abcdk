/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_SSL_EASYSSL_H
#define ABCDK_SSL_EASYSSL_H

#include "abcdk/util/general.h"
#include "abcdk/util/enigma.h"
#include "abcdk/util/object.h"

__BEGIN_DECLS

/** 简单的SSL通讯。 */
typedef struct _abcdk_easyssl abcdk_easyssl_t;


/**销毁。*/
void abcdk_easyssl_destroy(abcdk_easyssl_t **ctx);

/**创建。*/
abcdk_easyssl_t *abcdk_easyssl_create(const uint8_t *dict,size_t rotor);

/**
 * 关联句柄。
 * 
 * @return 旧的句柄。
*/
int abcdk_easyssl_set_fd(abcdk_easyssl_t *ctx,int fd);


__END_DECLS

#endif //ABCDK_SSL_EASYSSL_H


