/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_AUTH_AUTH_H
#define ABCDK_AUTH_AUTH_H

#include "abcdk-util/general.h"
#include "abcdk-util/getargs.h"
#include "abcdk-util/socket.h"
#include "abcdk-util/crc32.h"

__BEGIN_DECLS

/**/
#define ABCDK_AUTH_DEFAULT_KEY      123456789

/**/
#define ABCDK_AUTH_DEFAULT_MAGIC    987654321

/**
 * 添加DMI信息。
 * 
 * @return 0 成功，-1 失败(SN已经存在)，-2 失败(UUID已经存在)。
*/
int abcdk_auth_add_dmi(abcdk_tree_t *auth,const char *system_sn,const char * system_uuid);

/**
 * 添加MAC地址。
 * 
 * @return 0 成功，-1 失败(MAC已经存在)。
*/
int abcdk_auth_add_mac(abcdk_tree_t *auth,const char *mac);

/**
 * 添加有效期限。
 * 
 * @param days 使用天数。
 * @param begin 开始日期(UTC)，NULL(0) 使用当前时间。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_add_valid_period(abcdk_tree_t *auth, uintmax_t days, struct tm *begin);

/**
 * 添加有效期限。
 * 
 * @param days 使用天数。
 * @param delay 延时多少天开始(以当前时间(UTC)为基准)。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_add_valid_period2(abcdk_tree_t *auth, uintmax_t days, uintmax_t delay);

/**
 * 加盐。
 * 
 * @note 用于防止两次加密生成的密文相同。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_add_salt(abcdk_tree_t *auth);

/**
 * 收集DMI信息。
 * 
 * @note 仅包括system-serial-number和system-uuid。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_collect_dmi(abcdk_tree_t *auth);

/**
 * 收集MAC地址。
 * 
 * @note 仅包括物理地址。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_collect_mac(abcdk_tree_t *auth);

/**
 * 验证。
 * 
 * @return 0 成功，-1 失败(DMI或MAC不符合)，-2 失败(未在有效期限内)，-3 失败(其它)。
*/
int abcdk_auth_verify(abcdk_tree_t *auth);

/**
 * 序列化。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_allocator_t *abcdk_auth_serialize(abcdk_tree_t *auth);

/**
 * 结构化。
 * 
 * @param plaintext 序列化的授权信息。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_tree_t *abcdk_auth_structure(abcdk_allocator_t *plaintext);

/**
 * 加密。
 * 
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_allocator_t *abcdk_auth_encrypt(abcdk_allocator_t *plaintext, uint32_t key);

/**
 * 解密。
 * 
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_allocator_t *abcdk_auth_decrypt(abcdk_allocator_t *ciphertext, uint32_t key);

/**
 * 保存。
 * 
 * @warning 追加到文件末尾。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_save(int fd,const void *auth,size_t len,uint32_t magic);

/**
 * 保存。
 * 
 * @warning 追加到文件末尾。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_auth_save2(const char *file,const void *auth,size_t len,uint32_t magic);

/**
 * 加载。
 * 
 * @warning 从文件末尾加载。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_allocator_t *abcdk_auth_load(int fd,uint32_t magic);

/**
 * 加载。
 * 
 * @warning 从文件末尾加载。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_allocator_t *abcdk_auth_load2(const char *file,uint32_t magic);

__END_DECLS

#endif //ABCDK_AUTH_AUTH_H
