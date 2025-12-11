/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#include "abcdk/json/util.h"

#ifdef HAVE_JSONC

void _abcdk_json_format(FILE *fp,int better,size_t depth,json_object *obj)
{
    struct json_object_iterator it;
    struct json_object_iterator it_end;
    const char *key;
    json_object *sub;
    int list_size;
    const char *str_ptr;
    int str_len;
    json_type type;
    int items = 0;

    type = json_object_get_type(obj);

    if (type == json_type_int)
        fprintf(fp, "%lld", json_object_get_int64(obj));
    else if (type == json_type_boolean)
        fprintf(fp, "%llf", json_object_get_double(obj));
    else if (type == json_type_string)
    {
        str_ptr = json_object_get_string(obj);
        str_len = json_object_get_string_len(obj);
        if (better && str_len > 80)
            fprintf(fp, "\"%.80s[...+%dB]\"", str_ptr, str_len - 80);
        else
            fprintf(fp, "\"%s\"",str_ptr);
    }
    else if (type == json_type_boolean)
        fprintf(fp, "%s", json_object_get_boolean(obj) ? "true" : "false");
    else if (type == json_type_array)
    {
        fprintf(fp, "[\n");

        list_size = json_object_array_length(obj);
        for (size_t i = 0; i < list_size; i++)
        {
            if (i > 0)
                fprintf(fp, ",\n");

            for (size_t i = 0; i < depth + 1; i++)
                fprintf(fp, "\t");

            sub = json_object_array_get_idx(obj, i);
            _abcdk_json_format(fp, better, depth + 1, sub);
        }

        fprintf(fp, "\n");
        for (size_t i = 0; i < depth; i++)
            fprintf(fp, "\t");
        fprintf(fp, "]");
    }
    else if (type == json_type_object)
    {
        fprintf(fp, "{\n");

        it = json_object_iter_begin(obj);
        it_end = json_object_iter_end(obj);

        while (1)
        {
            if (json_object_iter_equal(&it, &it_end))
                break;

            if (++items > 1)
                fprintf(fp, ",\n");

            key = json_object_iter_peek_name(&it);
            sub = json_object_iter_peek_value(&it);
            type = json_object_get_type(sub);

            for (size_t i = 0; i < depth; i++)
                fprintf(fp, "\t");

            fprintf(fp, "\t\"%s\": ", key);
            _abcdk_json_format(fp, better, depth + 1, sub);

            json_object_iter_next(&it);
        }

        fprintf(fp, "\n");
        for (size_t i = 0; i < depth; i++)
            fprintf(fp, "\t");
        fprintf(fp, "}");
    }
    else
        fprintf(fp, "null");

}

#endif //#ifdef HAVE_JSONC

int abcdk_json_format_from_string(const char *str, size_t depth, int readable, FILE *out)
{
#ifndef HAVE_JSONC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含JSON-C工具。"));
    return -1;
#else //#ifndef HAVE_JSONC
    assert(str && out);

    json_object *ctx = json_tokener_parse(str);
    if(!ctx)
        return -1;

    _abcdk_json_format(out,readable,depth,ctx);
    json_object_put(ctx);

    return 0;
#endif //#ifndef HAVE_JSONC
}

int abcdk_json_format_from_file(const char *file, size_t depth, int readable, FILE *out)
{
#ifndef HAVE_JSONC
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含JSON-C工具。"));
    return -1;
#else //#ifndef HAVE_JSONC
    assert(file && out);

    json_object *ctx = json_object_from_file(file);
    if(!ctx)
        return -1;

    _abcdk_json_format(out,readable,depth,ctx);
    json_object_put(ctx);

    return 0;
#endif //#ifndef HAVE_JSONC
}
