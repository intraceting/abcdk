/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/option.h"

/** 选项。*/
struct _abcdk_option
{
    abcdk_tree_t *table;
};//abcdk_option_t;

/**
 * 选项的字段索引。
 */
typedef enum _abcdk_option_field
{
    /** Prefix。*/
    ABCDK_OPTION_PREFIX = 0,
#define ABCDK_OPTION_PREFIX ABCDK_OPTION_PREFIX

    /** Key。*/
    ABCDK_OPTION_KEY = 0,
#define ABCDK_OPTION_KEY ABCDK_OPTION_KEY

    /** Value。*/
    ABCDK_OPTION_VALUE = 0
#define ABCDK_OPTION_VALUE ABCDK_OPTION_VALUE

} abcdk_option_field_t;

abcdk_tree_t *_abcdk_option_find_key(abcdk_option_t *opt, const char *key, int create)
{
    abcdk_tree_t *it = NULL;
    const char *prefix;
    size_t pfx_len;
    char *in_key;
    int chk;

    prefix = opt->table->obj->pstrs[ABCDK_OPTION_PREFIX];
    pfx_len = strlen(opt->table->obj->pstrs[ABCDK_OPTION_PREFIX]);

    /*检测KEY是否包含前缀，如果不包含则添加。*/
    in_key = (char *)key;
    if (pfx_len > 0 && abcdk_strncmp(prefix, key, pfx_len, 0) != 0)
    {
        in_key = abcdk_heap_alloc(strlen(key) + 3);
        if (!in_key)
            return NULL;

        sprintf(in_key, "%s%s", prefix, key);
    }

    it = abcdk_tree_child(opt->table, 1);
    while (it)
    {
        chk = abcdk_strcmp(it->obj->pptrs[ABCDK_OPTION_KEY], in_key, 0);
        if (chk == 0)
            break;

        it = abcdk_tree_sibling(it, 0);
    }

    if (it == NULL && create != 0)
    {
        it = abcdk_tree_alloc3(strlen(in_key) + 1);

        if (it)
        {
            strcpy(it->obj->pptrs[ABCDK_OPTION_KEY], in_key);
            abcdk_tree_insert2(opt->table, it, 0);
        }
    }

    /*新创建要释放掉。*/
    if(in_key != key)   
        abcdk_heap_free(in_key);

    return it;
}

abcdk_tree_t *_abcdk_option_find_value(abcdk_tree_t *key,size_t index)
{
    abcdk_tree_t *it = NULL;
    size_t chk = 0;

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
abcdk_option_t *abcdk_option_alloc(const char *prefix)
{
    abcdk_option_t *opt = NULL;
    
    assert(prefix != NULL);

    opt = abcdk_heap_alloc(sizeof(abcdk_option_t));
    if (!opt)
        return NULL;

    opt->table = abcdk_tree_alloc3(strlen(prefix)+1);
    if (!opt->table)
        goto final_error;

    strcpy(opt->table->obj->pstrs[ABCDK_OPTION_PREFIX], prefix);

    return opt;

final_error:

    abcdk_option_free(&opt);

    return NULL;
}

const char *abcdk_option_prefix(abcdk_option_t *opt)
{
    assert(opt != NULL);

    return opt->table->obj->pstrs[ABCDK_OPTION_PREFIX];
}

int abcdk_option_set(abcdk_option_t *opt, const char *key, const char *value)
{
    abcdk_tree_t *it_key = NULL;
    abcdk_tree_t *it_val = NULL;

    assert(opt != NULL && key != NULL);
    assert(key[0] != '\0');

    it_key = _abcdk_option_find_key(opt,key,1);
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

int abcdk_option_fset(abcdk_option_t *opt, const char *key, const char *valfmt, ...)
{
    char value[4000] = {0};
    va_list ap;

    va_start(ap, valfmt);
    vsnprintf(value,4000,valfmt,ap);
    va_end(ap);

    return abcdk_option_set(opt,key,value);
}

const char* abcdk_option_get(abcdk_option_t *opt, const char *key,size_t index,const char* defval)
{
    abcdk_tree_t *it_key = NULL;
    abcdk_tree_t *it_val = NULL;

    assert(opt != NULL && key != NULL);
    assert(key[0] != '\0');

    it_key = _abcdk_option_find_key(opt,key,0);
    if(!it_key)
        return defval;

    it_val = _abcdk_option_find_value(it_key,index);
    if(!it_val)
        return defval;

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

    it_key = _abcdk_option_find_key(opt,key,0);
    if(!it_key)
        ABCDK_ERRNO_AND_RETURN1(EAGAIN,-1);

    return _abcdk_option_count_value(it_key);
}

int abcdk_option_remove(abcdk_option_t *opt, const char *key)
{
    abcdk_tree_t *it_key = NULL;

    assert(opt != NULL && key != NULL);
    assert(key[0] != '\0');

    it_key = _abcdk_option_find_key(opt,key,0);
    if(!it_key)
        ABCDK_ERRNO_AND_RETURN1(EAGAIN,-1);

    abcdk_tree_unlink(it_key);
    abcdk_tree_free(&it_key);

    return 0;
}

int _abcdk_option_scan_cb(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_option_iterator_t *it = (abcdk_option_iterator_t *)opaque;
    abcdk_tree_t *father;
    int chk;

    if (depth == 0 || depth == 1)
        return 1;
    else if (depth == SIZE_MAX || depth != 2)
        return -1;

    /*find key.*/
    father = abcdk_tree_father(node);

    chk = it->dump_cb(father->obj->pstrs[ABCDK_OPTION_KEY], node->obj->pstrs[ABCDK_OPTION_VALUE], it->opaque);
    if (chk < 0)
        return -1;

    return 1;
}

void abcdk_option_scan(abcdk_option_t *opt, abcdk_option_iterator_t *it)
{
    assert(opt != NULL && it != NULL);
    assert(it->dump_cb != NULL);

    abcdk_tree_iterator_t tit = {0,_abcdk_option_scan_cb, (void*)it};

    abcdk_tree_scan(opt->table, &tit);
}

int _abcdk_option_merge_merge_cb(const char *key,const char *value, void *opaque)
{
    abcdk_option_t *dst = (abcdk_option_t *)opaque;
    int chk;

    chk = abcdk_option_set(dst,key,value);
    if(chk != 0)
        return -1;

    return 1;
}

void abcdk_option_merge(abcdk_option_t *dst,abcdk_option_t *src)
{
    abcdk_option_iterator_t it;

    assert(dst != NULL && src != NULL);
    assert(dst != src);

    it.opaque = dst;
    it.dump_cb = _abcdk_option_merge_merge_cb;

    abcdk_option_scan(src,&it);
}