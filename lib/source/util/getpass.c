/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/getpass.h"

abcdk_object_t *abcdk_getpass(FILE *istream,const char *prompt,...)
{
    abcdk_object_t *passwd = NULL;
    FILE *in_fp = NULL;
    int in_fd = -1;
    struct termios in_old = {0};
    int chk,chk2;

    /*如果没有提示词，则使用默认的提示词。*/
    if (!prompt || !*prompt)
        return abcdk_getpass(istream, "Enter passphrase");

    va_list ap;
    va_start(ap, prompt);
    vfprintf(stdout, prompt, ap);
    va_end(ap);

    fprintf(stdout, ": ");

    /*立即显示。*/
    fflush(stdout);

    /*最大1024字符，包括结束符号。*/
    passwd = abcdk_object_alloc2(1024);
    if(!passwd)
        return NULL;

    in_fp = (istream ? istream : stdin);
    in_fd = fileno(in_fp);

    if (in_fd < 0)
        goto ERR;

    chk = abcdk_tcattr_cbreak(in_fd,&in_old);
    if(chk != 0)
        goto ERR;

    chk = 0;
    while (chk < (passwd->sizes[0] - 1))
    {
        chk2 = read(in_fd,&passwd->pstrs[0][chk],1);
        if(chk2 <= 0)
            goto ERR;

        if(passwd->pstrs[0][chk] == '\n')
            break;

        chk += 1;
    }
    
    /*fix T-NULL and length.*/
    passwd->pstrs[0][chk] = '\0';
    passwd->sizes[0] = chk;

    chk = abcdk_tcattr_option(in_fd,&in_old,NULL);
    if(chk != 0)
        goto ERR;

    /*换行。*/
    fprintf(stdout, "\n");

    return passwd;
ERR:

    abcdk_object_unref(&passwd);
    return NULL;
}