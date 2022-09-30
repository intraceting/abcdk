/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/html.h"

void _abcdk_html_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    if(alloc->pptrs[ABCDK_HTML_KEY])
        abcdk_heap_free(alloc->pptrs[ABCDK_HTML_KEY]);
    if(alloc->pptrs[ABCDK_HTML_VALUE])
        abcdk_heap_free(alloc->pptrs[ABCDK_HTML_VALUE]);
}

void _abcdk_html_attr_parse(abcdk_tree_t *tag, const char *b, const char *e)
{
    abcdk_tree_t *attr = NULL;
    const char *key_b = NULL;
    const char *key_e = NULL;
    const char *val_b = NULL;
    const char *val_e = NULL;
    char r = '\0';

    /**/
    key_b = b;

next:

    r = '\0';
    val_b = val_e = key_e = NULL;
    for (;; key_b++)
    {
        if (key_b == e)
            return;

        if (!isspace(*key_b))
            break;
    }

    if (*key_b == '>')
        return;

    if (*key_b == '/')
    {
        key_e = key_b + 1;
    }
    else
    {
        for (key_e = key_b;; key_e++)
        {
            if (key_e == e)
                goto copy_attr;

            /*特殊处理一下。*/
            if(abcdk_strcmp((char*)tag->alloc->pptrs[ABCDK_HTML_KEY],"!DOCTYPE",0)==0)
                continue;
            
            if (isspace(*key_e) || *key_e == '=' || *key_e == '>')
                break;
            
        }

        if (isspace(*key_e) || *key_e == '>')
            goto copy_attr;

        r = '\0';
        val_e = NULL;
        for (val_b = key_e;; val_b++)
        {
            if (val_b == e || *val_b == '>')
                goto copy_attr;

            if (!isspace(*val_b) && *val_b != '=')
                break;
        }

        /*检查VALUE是否补引号包围。*/
        if (*val_b == '\'' || *val_b == '\"')
        {
            r = *val_b;
            val_b += 1;
        }

        for (val_e = val_b;;val_e++)
        {
            if (val_e == e || *val_e == '>')
                goto copy_attr;
            
            /*
             * 1：引号包围要配对。
             * 2：其它以“空白字符”为标记。
            */
            if (r ? (*val_e == r) : isspace(*val_e))
                break;
        }
    }

copy_attr:

    attr = abcdk_tree_alloc2(NULL, 2, 0);
    if (!attr)
        return;

    /*复制VALUE。*/
    if (val_b && val_e && (val_e - val_b > 0))
        attr->alloc->pptrs[ABCDK_HTML_VALUE] = (uint8_t *)abcdk_heap_clone(val_b, val_e - val_b);

    /*复制KEY。*/
    attr->alloc->pptrs[ABCDK_HTML_KEY] = (uint8_t *)abcdk_heap_clone(key_b, key_e - key_b);

    /*注册专用内存回收函数。*/
    abcdk_object_atfree(attr->alloc, _abcdk_html_destroy_cb, NULL);

    /*加入到树的子节点末尾.*/
    abcdk_tree_insert2(tag,attr, 0);

    /*Next ATTR*/
    if(val_e)
        key_b = val_e + (r ? 1 : 0);
    else 
        key_b = key_e;

    goto next;
}

abcdk_tree_t *_abcdk_html_tag_parse(const char *b, const char *e)
{
    abcdk_tree_t *tag = NULL;
    const char *tmp = NULL;
    const char *key_b = NULL;
    const char *attr_b = NULL;

    tag = abcdk_tree_alloc2(NULL, 2, 0);
    if (!tag)
        goto final;

    /*也许是EOD标签。*/
    if (abcdk_strncmp(b, "</", 2, 0) == 0)
    {
        /*KEY首指针。*/
        key_b = tmp = b + 1;

        /*查找KEY尾指针。*/
        for (; tmp < e; tmp++)
        {
            if (*tmp == '>')
                break;
        }
        
        /*复制KEY。*/
        tag->alloc->pptrs[ABCDK_HTML_KEY] = (uint8_t *)abcdk_heap_clone(key_b, tmp - key_b);
    }
    else
    {
        if (*b == '<')
            tmp = b + 1;
        else
            tmp = b;

        /*KEY首指针。*/
        key_b = tmp;

        /*查找KEY尾指针。*/
        for (; tmp < e; tmp++)
        {
            if (*tmp == ' ' || *tmp == '>')
                break;
        }

        /*复制KEY。*/
        tag->alloc->pptrs[ABCDK_HTML_KEY] = (uint8_t *)abcdk_heap_clone(key_b, tmp - key_b);
        if (!tag->alloc->pptrs[ABCDK_HTML_KEY])
            goto final;

        /*也许已经到EOD.*/
        if (*tmp == '>')
            goto final;

        /*ATTR首指针。*/
        attr_b = tmp++;

        /*查找ATTR尾指针。*/
        for (; tmp < e; tmp++)
        {
            if (*tmp == '>')
                break;
        }

        /*分析ATTR。*/
        _abcdk_html_attr_parse(tag,attr_b,tmp);
    }

final:

    return tag;
}

const char *_abcdkc_html_cntrl_replace(char *text, char c)
{
    if(!text)
        return "";

    char *tmp = text;
    while (*tmp)
    {
        if (iscntrl(*tmp))
            *tmp = c;

        tmp += 1;
    }

    return text;
}

const char *_abcdk_html_value_parse(abcdk_tree_t *tag, const char *text)
{
    const char *tmp = NULL, *tmp2 = NULL;
    const char *value_b = NULL;

    /*也许是EOD标签。*/
    if (abcdk_strncmp(tag->alloc->pptrs[ABCDK_HTML_KEY], "/", 1, 0) == 0)
        return text;

    if (*text == '>')
        tmp = text + 1;
    else
        tmp = text;

    /*VALUE首指针。*/
    value_b = tmp;

    /*查找VALUE尾指针。*/
    if (abcdk_strncmp(tag->alloc->pptrs[ABCDK_HTML_KEY], "script", 6, 0) == 0)
    {
        /*脚本。*/
        tmp2 = abcdk_strstr(tmp, "</script>", 0);
        if (tmp2)
            tmp = tmp2;
    }
    else if (abcdk_strncmp(tag->alloc->pptrs[ABCDK_HTML_KEY], "style", 5, 0) == 0)
    {
        /*样式。*/
        tmp2 = abcdk_strstr(tmp, "</style>", 0);
        if (tmp2)
            tmp = tmp2;
    }
    else
    {
        /*普通文本。*/
        for (;; tmp++)
        {
            if (*tmp == '\0')
                goto final;

            if (*tmp == '<')
                break;
        }
    }

    /*可能没有VALUE。*/
    if (tmp - value_b <= 0)
        return tmp;

    /*复制VALUE。*/
    tag->alloc->pptrs[ABCDK_HTML_VALUE] = (uint8_t *)abcdk_heap_clone(value_b, tmp - value_b);
    
    /*替换控制字符为空格。*/
    _abcdkc_html_cntrl_replace((char*)tag->alloc->pptrs[ABCDK_HTML_VALUE],' ');

final:

    return tmp;
}

void _abcdk_html_parse_real(abcdk_tree_t *root, const char *text)
{
    abcdk_tree_t *tag = NULL;
    const char *b = NULL, *e = NULL;
    const char *tmp = NULL;

    tmp = text;

    while (tmp && *tmp)
    {
        e = NULL;
        b = abcdk_strstr(tmp, "<", 1);
        if (!b)
            break;

        if (abcdk_strncmp(b, "<!--", 4, 1) == 0)
        {
            tmp = abcdk_strstr_eod(b, "-->", 1);
        }
        else
        {
            e = abcdk_strstr_eod(b, ">", 1);
            if (!e)
                break;
            
            /*解析KEY和ATTR。*/
            tag = _abcdk_html_tag_parse(b, e);
            if (!tag)
                break;

            /*解析VALUE。*/
            tmp = _abcdk_html_value_parse(tag, e);

            /*注册专用内存回收函数。*/
            abcdk_object_atfree(tag->alloc,_abcdk_html_destroy_cb,NULL);

            /*加入到树的子节点末尾.*/
            abcdk_tree_insert2(root, tag, 0);
        }
    }
}

abcdk_tree_t *abcdk_html_parse_text(const char *text)
{
    abcdk_tree_t *root = NULL;

    assert(text != NULL);

    root = abcdk_tree_alloc3(1);
    if (!root)
        goto final;

    _abcdk_html_parse_real(root, text);

final:

    return root;
}

abcdk_tree_t *abcdk_html_parse_file(const char *file)
{
    abcdk_tree_t *root = NULL;
    abcdk_object_t *fmem = NULL;

    assert(file != NULL);

    fmem = abcdk_mmap2(file, 0, 0, 0);
    if (!fmem)
        goto final;

    root = abcdk_html_parse_text((char *)fmem->pptrs[0]);

final:

    abcdk_object_unref(&fmem);

    return root;
}