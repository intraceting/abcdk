/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/option.h"

/** 选项。*/
typedef struct _abcdk_option
{
    abcdk_tree_t *table;
};//abcdk_option_t;

/**
 * 选项的字段索引。
*/
typedef enum _abcdk_option_field
{
    /** Key。*/
   ABCDK_OPTION_KEY = 0,
#define ABCDK_OPTION_KEY     ABCDK_OPTION_KEY

    /** Value。*/
   ABCDK_OPTION_VALUE = 0
#define ABCDK_OPTION_VALUE   ABCDK_OPTION_VALUE

}abcdk_option_field_t;

abcdk_tree_t *_abcdk_option_find_key(abcdk_tree_t *opt, const char *key,int create)
{
    abcdk_tree_t *it = NULL;
    int chk;

    assert(opt != NULL && key != NULL);

    it = abcdk_tree_child(opt,1);
    while(it)
    {
        chk = abcdk_strcmp(it->obj->pptrs[ABCDK_OPTION_KEY], key, 1);
        if (chk == 0)
            break;

        it = abcdk_tree_sibling(it,0);
    }

    if(it == NULL && create !=0 )
    {
        it = abcdk_tree_alloc3(strlen(key)+1);

        if(it)
        {
            strcpy(it->obj->pptrs[ABCDK_OPTION_KEY],key);
            abcdk_tree_insert2(opt,it,0);
        }
    }

    return it;
}

abcdk_tree_t *_abcdk_option_find_value(abcdk_tree_t *key,size_t index)
{
    abcdk_tree_t *it = NULL;
    size_t chk = 0;

    assert(key != NULL);

    it = abcdk_tree_child(key,1);
    while(it)
    {
        if (chk++ == index)
            break;

        it = abcdk_tree_sibling(it,0);
    }

    return it;
}

size_t _abcdk_option_count_value(abcdk_tree_t *key)
{
    abcdk_tree_t *it = NULL;
    size_t chk = 0;

    assert(key != NULL);

    it = abcdk_tree_child(key,1);
    while(it)
    {
        chk +=1;

        it = abcdk_tree_sibling(it,0);
    }

    return chk;
}

void abcdk_option_free(abcdk_option_t **opt)
{
    abcdk_option_t *opt_p = NULL;

    if (!opt || !*opt)
        return;

    opt_p = *opt;
    *opt = NULL;

    abcdk_tree_free(&opt_p->table);
    abcdk_heap_free(opt_p);
}

/** 创建对象。*/
abcdk_option_t *abcdk_option_alloc()
{
    abcdk_option_t *opt = NULL;

    opt = abcdk_heap_alloc(sizeof(abcdk_option_t));
    if (!opt)
        return NULL;

    opt->table = abcdk_tree_alloc3(1);
    if (!opt->table)
        goto final_error;

    return opt;

final_error:

    abcdk_option_free(&opt);

    return NULL;
}

int abcdk_option_set(abcdk_option_t *opt, const char *key, const char *value)
{
    abcdk_tree_t *it_key = NULL;
    abcdk_tree_t *it_val = NULL;

    assert(opt != NULL && key != NULL);
    assert(key[0] != '\0');

    it_key = _abcdk_option_find_key(opt->table,key,1);
    if(!it_key)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM,-1);

    /*
     * 允许没有值。
    */
    if (value == NULL || value[0] == '\0')
        return 0;
    
    it_val = abcdk_tree_alloc3(strlen(value) + 1);
    if (!it_val)
        ABCDK_ERRNO_AND_RETURN1(ENOMEM, -1);

    strcpy(it_val->obj->pptrs[ABCDK_OPTION_VALUE], value);
    abcdk_tree_insert2(it_key, it_val, 0);
    

    return 0;
}

int abcdk_option_set2(abcdk_option_t *opt, const char *key, const char *value,int merge)
{
    const char *p;
    ssize_t count;
    int chk;

    assert(opt != NULL && key != NULL);

    if (!value || !merge)
        goto final;

    count = abcdk_option_count(opt, key);
    if (count <= 0)
        goto final;

    for (size_t i = 0; i < count; i++)
    {
        p = abcdk_option_get(opt, key, i, "");
        
        if (abcdk_strcmp(p, value, 1) == 0)
            return 0;
    }

final:

    return abcdk_option_set(opt, key, value);
}

const char* abcdk_option_get(abcdk_option_t *opt, const char *key,size_t index,const char* defval)
{
    abcdk_tree_t *it_key = NULL;
    abcdk_tree_t *it_val = NULL;

    assert(opt != NULL && key != NULL);
    assert(key[0] != '\0');

    it_key = _abcdk_option_find_key(opt->table,key,0);
    if(!it_key)
        ABCDK_ERRNO_AND_RETURN1(EAGAIN,defval);

    it_val = _abcdk_option_find_value(it_key,index);
    if(!it_val)
        ABCDK_ERRNO_AND_RETURN1(EAGAIN,defval);

    return it_val->obj->pptrs[ABCDK_OPTION_VALUE];
}

int abcdk_option_get_int(abcdk_option_t *opt, const char *key, size_t index, int defval)
{
    const char *val = abcdk_option_get(opt, key, index, NULL);

    if (!val)
        return defval;

    return strtol(val,NULL,0);
}


long abcdk_option_get_long(abcdk_option_t *opt, const char *key,size_t index,long defval)
{
    const char *val = abcdk_option_get(opt, key, index, NULL);

    if (!val)
        return defval;

    return strtol(val,NULL,0);
}

long long abcdk_option_get_llong(abcdk_option_t *opt, const char *key,size_t index,long long defval)
{
    const char *val = abcdk_option_get(opt, key, index, NULL);

    if (!val)
        return defval;

    return strtoll(val,NULL,0);
}

double abcdk_option_get_double(abcdk_option_t *opt, const char *key,size_t index,double defval)
{
    const char *val = abcdk_option_get(opt, key, index, NULL);

    if (!val)
        return defval;

    return atof(val);
}

ssize_t abcdk_option_count(abcdk_option_t *opt, const char *key)
{
    abcdk_tree_t *it_key = NULL;

    assert(opt != NULL && key != NULL);
    assert(key[0] != '\0');

    it_key = _abcdk_option_find_key(opt->table,key,0);
    if(!it_key)
        ABCDK_ERRNO_AND_RETURN1(EAGAIN,-1);

    return _abcdk_option_count_value(it_key);
}

int abcdk_option_remove(abcdk_option_t *opt, const char *key)
{
    abcdk_tree_t *it_key = NULL;

    assert(opt != NULL && key != NULL);
    assert(key[0] != '\0');

    it_key = _abcdk_option_find_key(opt->table,key,0);
    if(!it_key)
        ABCDK_ERRNO_AND_RETURN1(EAGAIN,-1);

    abcdk_tree_unlink(it_key);
    abcdk_tree_free(&it_key);

    return 0;
}

ssize_t abcdk_option_fprintf(abcdk_option_t *opt,FILE *fp,const char *hyphens)
{
    abcdk_tree_t *it_key = NULL, *it_val = NULL;
    const char* key = NULL, *val = NULL;
    ssize_t wsize = 0, wsize2 = 0;

    assert(fp != NULL && opt != NULL);
    assert(hyphens == NULL || hyphens[0] != '\0');

    it_key = abcdk_tree_child(opt->table,1);
    while(it_key)
    {
        key = (char*)it_key->obj->pptrs[ABCDK_OPTION_KEY];

        /*有连字符时不在这里输出。*/
        if (hyphens == NULL)
        {
            wsize2 = fprintf(fp, "%s\r\n", key);
            if (wsize2 <= 0)
                break;

            wsize += wsize2;
        }

        it_val = abcdk_tree_child(it_key,1);
        while(it_val)
        {
            val = (char*)it_val->obj->pptrs[ABCDK_OPTION_VALUE];

            /*无连字符时在这里输出。*/
            if (hyphens == NULL)
            {
                wsize2 = fprintf(fp, "%s\r\n", val);
                if (wsize2 <= 0)
                    break;

                wsize += wsize2;
            }
            else /*有连字符时在这里输出。*/
            {
                wsize2 = fprintf(fp, "%s%s%s\r\n",key,hyphens,val);
                if (wsize2 <= 0)
                    break;

                wsize += wsize2;
            }

            it_val = abcdk_tree_sibling(it_val,0);  
        }

        it_key = abcdk_tree_sibling(it_key,0);
    }

    return wsize;
}

ssize_t abcdk_option_snprintf(abcdk_option_t *opt,char* buf,size_t max,const char *hyphens)
{
    FILE* fp = NULL;
    ssize_t wsize = 0;

    assert(buf != NULL && max >0 && opt);

    fp = fmemopen(buf,max,"w");
    if(!fp)
        return -1;

    wsize = abcdk_option_fprintf(opt,fp,hyphens);

    fclose(fp);
    
    return wsize;
}
