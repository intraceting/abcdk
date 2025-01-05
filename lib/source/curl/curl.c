/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 * 
 */
#include "abcdk/curl/curl.h"

#ifdef CURLINC_CURL_H

static size_t _abcdk_curl_download_write_cb(void *buffer, size_t size, size_t nmemb, void *user_p)
{
    int *fd = (int *)user_p;

    ssize_t chk = abcdk_write(*fd, buffer, size * nmemb);
    if (chk > 0)
        return chk;

    return 0;
}

int abcdk_curl_download_fd(int fd,const char *url,size_t offset,size_t count,time_t ctimeout,time_t stimeout)
{
    CURL *curl_ctx = NULL;
    struct curl_slist *header_list = NULL;
    abcdk_object_t *url_en = NULL;
    char buf[100] = {0};
    long rspcode = 0;
    int chk;

    assert(fd >=0 && url != NULL);

    ctimeout = ABCDK_CLAMP(ctimeout,(time_t)1,(time_t)15);
    stimeout = ABCDK_CLAMP(stimeout,(time_t)5,(time_t)30);

    url_en = abcdk_url_encode2(url,strlen(url),0);
    if(!url_en)
        goto END;

    curl_ctx = curl_easy_init();
    if (!curl_ctx)
        goto END;

    header_list = curl_slist_append(header_list, "User-Agent: ABCDK (Linux;) libcurl/" LIBCURL_VERSION);
    curl_easy_setopt(curl_ctx, CURLOPT_HTTPHEADER, header_list);
    

    curl_easy_setopt(curl_ctx, CURLOPT_HEADER, 0);

    curl_easy_setopt(curl_ctx, CURLOPT_URL, url_en->pstrs[0]);

    curl_easy_setopt(curl_ctx, CURLOPT_SSL_VERIFYPEER, 0);
    curl_easy_setopt(curl_ctx, CURLOPT_SSL_VERIFYHOST, 0);

    curl_easy_setopt(curl_ctx, CURLOPT_VERBOSE, 0);

    curl_easy_setopt(curl_ctx, CURLOPT_READFUNCTION, NULL);
    curl_easy_setopt(curl_ctx, CURLOPT_WRITEFUNCTION, &_abcdk_curl_download_write_cb);
    curl_easy_setopt(curl_ctx, CURLOPT_WRITEDATA, &fd);

    curl_easy_setopt(curl_ctx, CURLOPT_NOSIGNAL, 1);

    curl_easy_setopt(curl_ctx, CURLOPT_CONNECTTIMEOUT, ctimeout);
    curl_easy_setopt(curl_ctx, CURLOPT_TIMEOUT, stimeout);

    if (count > 0)
        snprintf(buf, 100, "%zd-%zd", offset, count - 1);
    else if (offset > 0)
        snprintf(buf, 100, "%zd-", offset);

    if (strlen(buf) > 0)
        curl_easy_setopt(curl_ctx, CURLOPT_RANGE, buf);

    chk = curl_easy_perform(curl_ctx);

    curl_easy_getinfo(curl_ctx, CURLINFO_RESPONSE_CODE, &rspcode);


END:

    if(header_list)
        curl_slist_free_all(header_list);
    if(curl_ctx)
        curl_easy_cleanup(curl_ctx);
    abcdk_object_unref(&url_en);

    abcdk_trace_printf(LOG_DEBUG,"++++++++\nsrc: '%s'\nrsp: %ld\nchk: '%s'\n--------\n", url, rspcode, curl_easy_strerror(chk));

    if (chk == CURLE_OK && (rspcode == 200||rspcode == 206))
        return 0;

    return -1;
}

int abcdk_curl_download_filename(const char *file,const char *url,size_t offset,size_t count,time_t ctimeout,time_t stimeout)
{
    int fd;
    int chk;

    assert(file >=0 && url != NULL);

    fd = abcdk_open(file,1,0,1);
    if(fd <0)
        return -1;

    chk = abcdk_curl_download_fd(fd,url,offset,count,ctimeout,stimeout);
    abcdk_closep(&fd);

    return chk;
}


#endif //CURLINC_CURL_H