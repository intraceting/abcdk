/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#include "abcdk/json/util.h"

#ifdef _json_h_

void abcdk_json_readable(FILE *fp,int better,size_t depth,json_object *obj)
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
            fprintf(fp, "\"(len=%d)%.80s\"",str_len,str_ptr);
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
            abcdk_json_readable(fp, better, depth + 1, sub);
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
            abcdk_json_readable(fp, better, depth + 1, sub);

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

void abcdk_json_unref(json_object **obj)
{
    int chk;
    
    if(!obj || !*obj)
        return;

    chk = json_object_put(*obj);
    assert(chk == 1);

    /*Set to NULL(0).*/
    *obj = NULL;
}

json_object *abcdk_json_refer(json_object *obj)
{
    assert(obj != NULL);

    return json_object_get(obj);
}

json_object* abcdk_json_locate(json_object *father,...)
{
    const char *key_p = NULL;
    json_object *prev = NULL,*next = NULL;
    json_bool chk = 0;

    assert(father != NULL);
    
    va_list vaptr;
    va_start(vaptr, father);

    for (prev = father;;prev = next)
    {
        key_p = va_arg(vaptr, const char *);
        if (!key_p)
            break;

        next = NULL;
        chk = json_object_object_get_ex(prev,key_p,&next);
        if(!chk)
            break;
    }

    va_end(vaptr);

    return next;
}

json_object *abcdk_json_parse(const char *str)
{
    assert(str != NULL);

    return json_tokener_parse(str);
}

const char *abcdk_json_string(json_object *obj)
{
    assert(obj != NULL);

    return json_object_to_json_string(obj);
}

void abcdk_json_add(json_object *father, const char *key, json_object *val)
{
    assert(father != NULL && key != NULL && val != NULL);
    assert(key[0] != '\0');

    /*不会改变子节点的引用计数。*/
    json_object_object_add(father, key, val);
}

json_object *abcdk_json_add_vformat(json_object *father, const char *key, const char *val_fmt, va_list val_args)
{
    json_object *sub = NULL;
    char buf[1024] = {0};

    assert(val_fmt != NULL && val_args != NULL);

    vsnprintf(buf, 1024, val_fmt, val_args);
    sub = json_object_new_string(buf);
    if (!sub)
        return NULL;

    if (father && key)
        abcdk_json_add(father, key, sub);

    return sub;
}

json_object *abcdk_json_add_format(json_object *father,const char *key, const char *val_fmt, ...)
{
    json_object *sub = NULL;

    va_list vaptr;
    va_start(vaptr, val_fmt);

    sub = abcdk_json_add_vformat(father,key,val_fmt, vaptr);

    va_end(vaptr);

    return sub;
}

json_object *abcdk_json_add_int32(json_object *father,const char *key,int32_t val)
{
    json_object *sub = NULL;

    sub = json_object_new_int(val);
    
    if (father && key)
        abcdk_json_add(father, key, sub);

    return sub;
}

json_object *abcdk_json_add_int64(json_object *father,const char *key,int64_t val)
{
    json_object *sub = NULL;

    sub = json_object_new_int64(val);
    
    if (father && key)
        abcdk_json_add(father, key, sub);

    return sub;
}

json_object *abcdk_json_add_boolean(json_object *father,const char *key,json_bool val)
{
    json_object *sub = NULL;

    sub = json_object_new_boolean(val);
    
    if (father && key)
        abcdk_json_add(father, key, sub);

    return sub;
}

json_object* abcdk_json_add_double(json_object *father,const char *key,double val)
{
    json_object *sub = NULL;

    sub = json_object_new_double(val);
    
    if (father && key)
        abcdk_json_add(father, key, sub);

    return sub;
}



#endif //_json_h_
