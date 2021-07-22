/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "uri.h"

abcdk_allocator_t *abcdk_uri_split(const char *uri)
{
    const char* mark = NULL;
    const char* a_mark = NULL;
    size_t len = 0;
    size_t sizes[5] = {0};
    abcdk_allocator_t *alloc = NULL;

    assert(uri != NULL);

    assert(uri[0] != '\0');

    len = strlen(uri);

    mark = abcdk_strstr(uri,"://",0);
    if (mark)
    {
        sizes[ABCDK_URI_SCHEME] = 64;
        sizes[ABCDK_URI_USER] = 64;
        sizes[ABCDK_URI_PSWD] = 128;
        sizes[ABCDK_URI_HOST] = 255;
        sizes[ABCDK_URI_PATH] = len + 1;
    }
    else
    {
        sizes[ABCDK_URI_SCHEME] = sizes[ABCDK_URI_USER] = 1;
        sizes[ABCDK_URI_PSWD] = sizes[ABCDK_URI_HOST] = 1;
        sizes[ABCDK_URI_PATH] = len + 1; //set.
    }

    alloc = abcdk_allocator_alloc(sizes,ABCDK_ARRAY_SIZE(sizes),0);
    if(!alloc)
        goto final;
    
    if(mark)
    {
        if(ABCDK_PTR2I8(mark,3) == '/')
        {
            /* SCHEME:///abcdk/...*/
            sscanf(uri,"%[^:]%*3[:/]%s",alloc->pptrs[ABCDK_URI_SCHEME],alloc->pptrs[ABCDK_URI_PATH]);
        }
        else
        {
            for (size_t k = 3; ABCDK_PTR2I8(mark, k); k++)
            {
                if (ABCDK_PTR2I8(mark, k) == '/')
                    break;

                if (ABCDK_PTR2I8(mark, k) == '@')
                    a_mark = ABCDK_PTR2I8PTR(mark, k);
            }

            if(a_mark)
            {
                /* 
                 * SCHEME://user:pswd@host[:port]/abcdk
                 * SCHEME://user:pswd@[host][:port]/abcdk
                */
                sscanf(uri, "%[^:]%*3[:/]%[^:]%*1[:]%[^@]%*1[@]%[^/]%s",
                       alloc->pptrs[ABCDK_URI_SCHEME], alloc->pptrs[ABCDK_URI_USER],
                       alloc->pptrs[ABCDK_URI_PSWD], alloc->pptrs[ABCDK_URI_HOST],
                       alloc->pptrs[ABCDK_URI_PATH]);
            }
            else
            {
                /* 
                 * SCHEME://host[:port]/abcdk
                 * SCHEME://[host][:port]/abcdk
                */
                sscanf(uri, "%[^:]%*3[:/]%[^/]%s",
                       alloc->pptrs[ABCDK_URI_SCHEME], alloc->pptrs[ABCDK_URI_HOST],
                       alloc->pptrs[ABCDK_URI_PATH]);
            }
        }
    }
    else
    {
        memcpy(alloc->pptrs[ABCDK_URI_PATH],uri,len);
    }

final:

    return alloc;
}