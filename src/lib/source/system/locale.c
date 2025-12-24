/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/system/locale.h"

int abcdk_locale_setup(const char *lang_codeset, const char *domain_name, const char *domain_path)
{
    const char *tmp_lang_codeset_p = NULL;
    char tmp_domain_name[NAME_MAX] = {0};
    char tmp_domain_path[PATH_MAX] = {0};
    int chk;

    /*如果未指定语言和编码, 则从环境变获取语言配置.*/
    tmp_lang_codeset_p = (lang_codeset ? lang_codeset : getenv("ABCDK_DEFAULT_LANG_CODESET"));

    /*设置语言环境.*/
    setlocale(LC_ALL, (tmp_lang_codeset_p ? tmp_lang_codeset_p : "zh_CN.UTF-8"));

    if(!domain_name)
        abcdk_proc_basename(tmp_domain_name);//未指定域名, 以当前程序为域名.
    else 
        strncpy(tmp_domain_name,domain_name,NAME_MAX);
    
    if (!domain_path)
        abcdk_proc_dirname(tmp_domain_path, "../share/locale/");//未指定路径, 则以使用默认路径.
    else if (*domain_path == '/')
        strncpy(tmp_domain_path,domain_path,PATH_MAX);//绝对路径直接用.
    else
        abcdk_proc_dirname(tmp_domain_path, domain_path);//相对路径, 则以当前程序为父级路径.

    /*绑定域名和路径.*/
    chk = bindtextdomain(tmp_domain_name, tmp_domain_path);
    if (chk != 0)
        return -2;

    /*激活域名.*/
    chk = textdomain(tmp_domain_name);
    if (chk != 0)
        return -3;

    return 0;
}