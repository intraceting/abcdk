/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/getargs.h"

void abcdk_getargs(abcdk_option_t *opt, int argc, char *argv[])
{
    const char *prefix = NULL;
    size_t prefix_len = 0;
    const char *it_key = NULL;

    assert(opt != NULL && argc > 0 && argv != NULL);
    assert(argv[0] != NULL && argv[0][0] != '\0');

    prefix = abcdk_option_prefix(opt);
    prefix_len = strlen(prefix);
    it_key = prefix;

    for (int i = 0; i < argc;)
    {
        if (abcdk_strncmp(argv[i], prefix, prefix_len, 1) != 0)
        {
            abcdk_option_set(opt, it_key, argv[i++]);
        }
        else
        {
            abcdk_option_set(opt, it_key = argv[i++], NULL);
        }
    }
}

void abcdk_getargs_fp(abcdk_option_t *opt, FILE *fp, uint8_t delim, char note,const char *argv0)
{
    const char *prefix = NULL;
    size_t prefix_len = 0;
    const char *it_key = NULL;
    char *line = NULL;
    size_t len = 0;
    ssize_t rlen = 0;
    size_t rows = 0;
    char *key_p = NULL;
    char *val_p = NULL;

    assert(opt != NULL && fp != NULL);

    prefix = abcdk_option_prefix(opt);
    prefix_len = strlen(prefix);
    it_key = prefix;

    if (argv0)
        abcdk_option_set(opt, it_key, argv0);

    while (1)
    {
        rlen = abcdk_fgetline(fp, &line, &len, delim, note);
        if(rlen < 0)
            break;

        /* 去掉字符串两端所有空白字符。 */
        abcdk_strtrim(line, isspace, 2);

        if (abcdk_strncmp(line, prefix, prefix_len, 1) != 0)
        {
            abcdk_option_set(opt, it_key, line);
        }
        else
        {
            if (it_key != prefix)
                abcdk_heap_freep((void **)&it_key);

            it_key = abcdk_heap_clone(line, rlen);
            if (!it_key)
                break;

            abcdk_option_set(opt, it_key, NULL);
        }
    }

    /*不要忘记释放这两块内存，不然可能会有内存泄漏的风险。 */
    if (line)
        free(line);
    if (it_key != prefix)
        abcdk_heap_freep((void **)&it_key);
}

void abcdk_getargs_file(abcdk_option_t *opt, const char *file, uint8_t delim, char note, const char *argv0)
{
    FILE *fp = NULL;

    assert(opt != NULL && file != NULL);

    fp = fopen(file, "r");
    if (!fp)
        return;

    abcdk_getargs_fp(opt, fp, delim, note, argv0);

    fclose(fp);
}

void abcdk_getargs_text(abcdk_option_t *opt, const char *text, size_t len, uint8_t delim,char note, const char *argv0)
{
    FILE *fp = NULL;

    assert(opt != NULL && text != NULL && len > 0);

    fp = fmemopen((char *)text, len, "r");
    if (!fp)
        return;

    abcdk_getargs_fp(opt, fp, delim, note, argv0);

    fclose(fp);
}

typedef struct _abcdk_getargs_fprintf_param
{
    const char *delim;
    const char *pack;
    ssize_t wlen;
    FILE *fp;
    const char *prev_key;
} abcdk_getargs_fprintf_param_t;

int _abcdk_getargs_scan_cb(const char *key, const char *value, void *opaque)
{
    abcdk_getargs_fprintf_param_t *p = (abcdk_getargs_fprintf_param_t *)opaque;
    ssize_t wlen = 0;

    if (p->prev_key != key)
    {
        wlen = fprintf(p->fp, "%s%s%s%s", p->pack, key, p->pack, p->delim); //  包装，KEY，包装，分割符。
        if (wlen <= 0)
            return -1;

        p->prev_key = key;
        p->wlen += wlen;
    }

    wlen = fprintf(p->fp, "%s%s%s%s", p->pack, value, p->pack, p->delim); // 包装，KEY，包装，分割符。
    if (wlen <= 0)
        return -1;

    p->wlen += wlen;

    return 1;
}

ssize_t abcdk_getargs_fprintf(abcdk_option_t *opt,FILE *fp, const char *delim,const char *pack)
{
    abcdk_getargs_fprintf_param_t p;
    abcdk_option_iterator_t it;

    assert(opt != NULL && fp != NULL && delim != NULL && pack != NULL);

    p.delim = delim;
    p.pack = pack;
    p.fp = fp;
    p.prev_key = NULL;
    p.wlen = 0;

    it.dump_cb = _abcdk_getargs_scan_cb;
    it.opaque = &p;

    abcdk_option_scan(opt,&it);

    return p.wlen;
}

ssize_t abcdk_getargs_snprintf(abcdk_option_t *opt, char *buf, size_t max, const char *delim, const char *pack)
{
    FILE *fp = NULL;
    ssize_t wsize = 0;

    assert(opt != NULL && buf != NULL && max > 0 && delim != NULL && pack != NULL);

    fp = fmemopen(buf, max, "w");
    if (!fp)
        return -1;

    wsize = abcdk_getargs_fprintf(opt,fp,delim,pack);

    fclose(fp);

    return wsize;
}
