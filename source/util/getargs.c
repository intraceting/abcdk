/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/getargs.h"

void abcdk_getargs(abcdk_tree_t *opt, int argc, char *argv[],
                   const char *prefix)
{
    size_t prefix_len = 0;
    const char *it_key = NULL;

    assert(opt != NULL && argc > 0 && argv != NULL && prefix != NULL);

    assert(argv[0] != NULL && argv[0][0] != '\0' && prefix[0] != '\0');

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

void abcdk_getargs_fp(abcdk_tree_t *opt, FILE *fp, uint8_t delim, char note,
                      const char *argv0, const char *prefix)
{
    size_t prefix_len = 0;
    const char *it_key = NULL;
    char *line = NULL;
    size_t len = 0;
    size_t rows = 0;
    char *key_p = NULL;
    char *val_p = NULL;

    assert(opt != NULL && fp != NULL && prefix != NULL);

    prefix_len = strlen(prefix);
    it_key = prefix;

    if (argv0)
        abcdk_option_set(opt, it_key, argv0);

    while (abcdk_getline(fp, &line, &len, delim, note) != -1)
    {
        /* 去掉字符串两端所有空白字符。 */
        abcdk_strtrim(line, isspace, 2);

        if (abcdk_strncmp(line, prefix, prefix_len, 1) != 0)
        {
            abcdk_option_set(opt, it_key, line);
        }
        else
        {
            if (it_key != prefix)
                abcdk_heap_free2((void **)&it_key);

            it_key = abcdk_heap_clone(line, len + 1);
            if (!it_key)
                break;

            abcdk_option_set(opt, it_key, NULL);
        }
    }

    /*不要忘记释放这两块内存，不然可能会有内存泄漏的风险。 */
    if (line)
        free(line);
    if (it_key != prefix)
        abcdk_heap_free2((void **)&it_key);
}

void abcdk_getargs_file(abcdk_tree_t *opt, const char *file, uint8_t delim, char note,
                        const char *argv0, const char *prefix)
{
    FILE *fp = NULL;

    assert(opt != NULL && file != NULL);

    fp = fopen(file, "r");
    if (!fp)
        return;

    abcdk_getargs_fp(opt, fp, delim, note, argv0, prefix);

    fclose(fp);
}

void abcdk_getargs_text(abcdk_tree_t *opt, const char *text, size_t len, uint8_t delim, char note,
                        const char *argv0, const char *prefix)
{
    FILE *fp = NULL;

    assert(opt != NULL && text != NULL && len > 0);

    fp = fmemopen((char *)text, len, "r");
    if (!fp)
        return;

    abcdk_getargs_fp(opt, fp, delim, note, argv0, prefix);

    fclose(fp);
}