/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_TEST_ENTRY_H
#define ABCDK_TEST_ENTRY_H

#include "util/general.h"
#include "util/option.h"
#include "util/getargs.h"
#include "shell/proc.h"
#include "util/openssl.h"

__BEGIN_DECLS

int abcdk_test_http(abcdk_tree_t *args);
int abcdk_test_uri(abcdk_tree_t *args);
int abcdk_test_log(abcdk_tree_t *args);
int abcdk_test_easy(abcdk_tree_t *args);
int abcdk_test_iconv(abcdk_tree_t *args);

__END_DECLS

#endif //ABCDK_TEST_ENTRY_H
