/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/getpass.h"

abcdk_object_t *abcdk_getpass(const char *prompt, FILE *istream)
{
    abcdk_object_t *passwd = NULL;
    FILE *in_fp = NULL;
    int in_fd = -1;
    struct termios in_old = {0};
    int chk,chk2;

    passwd = abcdk_object_alloc2(1024);
    if(!passwd)
        return NULL;

    if (prompt && *prompt)
        fprintf(stdout, "%s: ", prompt);
    else
        fprintf(stdout, "Enter password: ");

    /*立即显示。*/
    fflush(stdout);

    in_fp = (istream ? istream : stdin);
    in_fd = fileno(in_fp);

    if (in_fd < 0)
        goto ERR;
    
    /*清除缓冲区内所有字符。*/
    //fscanf(in_fp,"%*[^\n]%*[^\n]");

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

    return passwd;
ERR:

    abcdk_object_unref(&passwd);
    return NULL;
}