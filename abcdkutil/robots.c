/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "robots.h"

abcdk_tree_t *_abcdk_robots_parse_rule(const char *line)
{
    abcdk_tree_t *rule = NULL;
    const char *key_b = NULL,*key_e = NULL;
    const char *val_b = NULL,*val_e = NULL;

    /* 查找KEY开始。*/
    for (key_b = line;;key_b++)
    {
        if (*key_b == '\0')
            ABCDK_ERRNO_AND_GOTO1(EPERM, final);

        if (!isspace(*key_b))
            break;
    }
    
    /* 查找KEY结束。*/
    for (key_e = key_b;;key_e++)
    {
        if (*key_e == '\0')
            ABCDK_ERRNO_AND_GOTO1(EPERM, final);

        if (isspace(*key_e) || *key_e == ':')
            break;
    }

    for (val_b = key_e;;val_b++)
    {
        if (*val_b == '\0')
            ABCDK_ERRNO_AND_GOTO1(EPERM, final);

        if (*val_b == ':')
            break;
    }

    /* 跳过连字符。*/
    if (*val_b == ':')
        val_b += 1;

    /* 查找VALUE开始。*/
    for (;;val_b++)
    {
        if (*val_b== '\0')
            ABCDK_ERRNO_AND_GOTO1(EPERM, final);

        if(!isspace(*val_b))
            break;
    }

    /*查找VALUE结束。*/
    for (val_e = val_b + 1;; val_e++)
    {
        if (*val_e == '\0')
            break;
    }

    size_t sizes[2] = {(key_e - key_b) + 1, (val_e - val_b) + 1};

    rule = abcdk_tree_alloc2(sizes,2,0);
    if(!rule)
        goto final;

    strncpy((char *)rule->alloc->pptrs[ABCDK_ROBOTS_KEY], key_b, key_e - key_b);
    strncpy((char *)rule->alloc->pptrs[ABCDK_ROBOTS_VALUE], val_b, val_e - val_b);

final:

    return rule;
}

void _abcdk_robots_parse_real(abcdk_tree_t *root, const char *text, const char *agent)
{
    abcdk_tree_t *rule = NULL;
    abcdk_buffer_t *buf = NULL;
    char *line = NULL;
    size_t size = 16 * PATH_MAX;
    ssize_t size2 = 0;
    int agent_ok = 0;

    buf = abcdk_buffer_alloc(NULL);
    if(!buf)
        return;

    buf->data = (void*)text;
    buf->size = strlen(text);
    buf->wsize = buf->size;
    buf->rsize = 0;

    line = abcdk_heap_alloc(size);
    if(!line)
        goto final;

    for(;;)
    {
        size2 = abcdk_buffer_readline(buf, line, size, '\n');
        if (size2 <= 0)
            break;

        /* 跳过太长的。*/
        if (size2 == size && line[size2 - 1] != '\n')
            continue;

        /* 去掉字符串两端所有空白字符。 */
        abcdk_strtrim(line, isspace, 2);

        /* 可能是注释。*/
        if (*line == '#')
            continue;

        /* 解析成KEY-VALUE节点。*/
        rule = _abcdk_robots_parse_rule(line);
        if(!rule)
            continue;

        /* 是否为段落标记。*/
        if(abcdk_strcmp(ABCDK_PTR2I8PTR(rule->alloc->pptrs[ABCDK_ROBOTS_KEY], 0),"User-agent",0)==0)
        {
            /* 是否为新段落。*/
            agent_ok = (abcdk_strcmp(ABCDK_PTR2I8PTR(rule->alloc->pptrs[ABCDK_ROBOTS_VALUE], 0),agent,0)==0);
        }

        /* 如果未找到匹配的段落则跳过。*/
        if(!agent_ok)
            continue;
      
        /*加入到树的子节点末尾.*/
        abcdk_tree_insert2(root, rule, 0);

    }

final:

    abcdk_heap_free(line);
    abcdk_buffer_free(&buf);
}

abcdk_tree_t *abcdk_robots_parse_text(const char *text,const char *agent)
{
    abcdk_tree_t *root = NULL;

    assert(text != NULL);

    root = abcdk_tree_alloc3(1);
    if (!root)
        goto final;

    _abcdk_robots_parse_real(root, text, agent);

final:

    return root;
}

abcdk_tree_t *abcdk_robots_parse_file(const char *file,const char *agent)
{
    abcdk_tree_t *root = NULL;
    abcdk_allocator_t *fmem = NULL;

    assert(file != NULL);

    fmem = abcdk_mmap2(file, 0, 0);
    if (!fmem)
        goto final;

    root = abcdk_robots_parse_text((char *)fmem->pptrs[0],agent);

final:

    abcdk_allocator_unref(&fmem);

    return root;
}