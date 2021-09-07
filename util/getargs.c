/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-util/getargs.h"

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

ssize_t _abcdk_getargs_getline(FILE *fp, char **line, size_t *len, uint8_t delim, char note)
{
    char *line_p = NULL;
    ssize_t chk = -1;

    while ((chk = getdelim(line, len, delim, fp)) != -1)
    {
        line_p = *line;

        if (*line_p == '\0' || *line_p == note || iscntrl(*line_p))
            continue;
        else
            break;
    }

    return chk;
}

int _abcdk_getargs_valtrim(int c)
{
    return (iscntrl(c) || (c == '\"') || (c == '\''));
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

    assert(opt != NULL && fp != NULL);

    if (prefix != NULL)
    {
        prefix_len = strlen(prefix);
        it_key = prefix;

        if (argv0)
            abcdk_option_set(opt, it_key, argv0);

        while (_abcdk_getargs_getline(fp, &line, &len, delim, note) != -1)
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
    }
    else
    {
        while (_abcdk_getargs_getline(fp, &line, &len, delim, note) != -1)
        {
            /* Find key.*/
            key_p = line;

            /* Find Value.*/
            val_p = strchr(line, '=');
            if (!val_p)
                val_p = strchr(line, ':');

            if (val_p)
            {
                *val_p = '\0'; // for key end.
                val_p += 1;

                /* 去掉value两端所有控制字符、双引号、单引号。 */
                abcdk_strtrim(val_p, _abcdk_getargs_valtrim, 2);
            }

            /* 去掉key两端所有空白字符。 */
            abcdk_strtrim(key_p, isspace, 2);

            abcdk_option_set(opt, key_p, val_p);
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