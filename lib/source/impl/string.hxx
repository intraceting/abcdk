/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_IMPL_STRING_HXX
#define ABCDK_IMPL_STRING_HXX

#include "abcdk/util/string.h"
#include "general.hxx"

#include <string>
#include <vector>

namespace abcdk
{
    namespace string
    {
        ABCDK_INVOKE_HOST void split(std::vector<std::string> &strs, const char *str, const char *delim)
        {
            const char *next_p;
            abcdk_object_t *p;

            assert(str != NULL && delim != NULL);
            assert(*delim != '\0');

            next_p = str;

            while (1)
            {
                p = abcdk_strtok3(&next_p, delim, 1);
                if (!p)
                    break;

                strs.push_back(p->pstrs[0]);
            }
        }

        ABCDK_INVOKE_HOST void split(std::vector<std::vector<std::string>> &strs, const char *str, const char *delim1, const char *delim2)
        {
            std::vector<std::string> tmps;

            assert(str != NULL && delim1 != NULL && delim2 != NULL);
            assert(*delim1 != '\0' || *delim2 != '\0');

            split(tmps, str, delim1);

            strs.resize(tmps.size());
            for (int i = 0; i < tmps.size(); i++)
                split(strs[i], tmps[i].c_str(), delim2);
        }
    } // namespace string

} // namespace abcdk

#endif // ABCDK_IMPL_STRING_HXX