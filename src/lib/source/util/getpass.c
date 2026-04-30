/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/general.h"
#include "abcdk/util/termios.h"
#include "abcdk/util/object.h"
#include "abcdk/util/getpass.h"

abcdk_object_t *abcdk_getpass(FILE *fp, const char *prompt, ...)
{
    abcdk_object_t *passwd = NULL;
    int in_fd = -1;
    struct termios old_cfg = {0};
    int chk, chk2;

    assert(fp != NULL);

    /*如果没有提示词, 则使用默认的提示词.*/
    if (!prompt || !*prompt)
        return abcdk_getpass(fp, "Enter passphrase");

    va_list ap;
    va_start(ap, prompt);
    vfprintf(stdout, prompt, ap);
    va_end(ap);

    fprintf(stdout, ": ");

    /*立即显示.*/
    fflush(stdout);

    /*最大1024字符, 包括结束符号.*/
    passwd = abcdk_object_alloc2(1024);
    if (!passwd)
        return NULL;

    in_fd = fileno(fp);

    if (in_fd < 0)
        goto ERR;

    chk = abcdk_tcattr_cbreak(in_fd, &old_cfg);
    if (chk != 0)
        goto ERR;

    chk = 0;
    while (chk < (passwd->sizes[0] - 1))
    {
        chk2 = read(in_fd, &passwd->pstrs[0][chk], 1);
        if (chk2 <= 0)
            goto ERR;

        if (passwd->pstrs[0][chk] == '\n')
            break;

        chk += 1;
    }

    /*fix T-NULL and length.*/
    passwd->pstrs[0][chk] = '\0';
    passwd->sizes[0] = chk;

    chk = abcdk_tcattr_option(in_fd, &old_cfg, NULL);
    if (chk != 0)
        goto ERR;

    /*换行.*/
    fprintf(stdout, "\n");

    return passwd;
ERR:

    abcdk_object_unref(&passwd);
    return NULL;
}

int abcdk_get_password(char *buf, int size, int enc_or_dec, void *opaque)
{
    abcdk_object_t *pass_a = NULL;
    abcdk_object_t *pass_b = NULL;
    int pass_len = 0;

    pass_a = abcdk_getpass(stdin, ABCDK_GETTEXT("输入密钥(%s)"), opaque);
    if (!pass_a)
        return -ENOMEM;

    if (size < pass_a->sizes[0])
    {
        fprintf(stdout, ABCDK_GETTEXT("密钥长度需小于%d个字符.\n"), size);
        return -1;
    }

    pass_len = pass_a->sizes[0]; // 密钥有效长度.

    if (enc_or_dec)
    {
        pass_b = abcdk_getpass(stdin, ABCDK_GETTEXT("验证密钥"));
        if (!pass_b)
            return -ENOMEM;

        if ((pass_len != pass_b->sizes[0]) ||
            memcmp(pass_a->pptrs[0], pass_b->pptrs[0], pass_len) != 0)
        {
            fprintf(stdout, ABCDK_GETTEXT("验证失败.\n"));
            return -1;
        }
    }

    memcpy(buf, pass_a->pptrs[0], pass_len);
    return pass_len;
}