/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/enigma.h"

/** Enigma转子。 */
typedef struct _abcdk_enigma_rotor
{
    /** 正向字典。*/
    uint8_t fdict[256];
    /** 逆向字典。*/
    uint8_t bdict[256];
    /** 步进指针。*/
    uint8_t pos;

} abcdk_enigma_rotor_t;

/** Enigma加密机。 */
struct _abcdk_enigma
{
    /** 转子数组。*/
    abcdk_enigma_rotor_t *rotots;

    /** 配置。*/
    abcdk_enigma_config_t *cfg;

}; // abcdk_enigma_t;

void abcdk_enigma_free(abcdk_enigma_t **ctx)
{
    abcdk_enigma_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_heap_free2((void **)&ctx_p->rotots);
    abcdk_heap_free2((void **)&ctx_p->cfg);

    abcdk_heap_free(ctx_p);
}

abcdk_enigma_t *abcdk_enigma_create(abcdk_enigma_config_t *cfg)
{
    abcdk_enigma_t *ctx = NULL;
    abcdk_enigma_rotor_t *rotor = NULL;
    uint8_t chk_dict[256 / 8];
    int chk;

    assert(cfg != NULL);
    assert(cfg->count > 0);

    /*检查字典表，每张字典表中的字符不能出现重复的。*/
    for (uint8_t y = 0; y < cfg->count; y++)
    {
        memset(chk_dict, 0, 256 / 8);
        for (uint8_t x = 0; x < 256; x++)
        {
            chk = abcdk_bloom_mark(chk_dict, 256 / 8, cfg->dict[y][x]);
            if (chk)
                return NULL;
        }
    }

    ctx = abcdk_heap_alloc(sizeof(abcdk_enigma_t));
    if (!ctx)
        return NULL;

    ctx->cfg = abcdk_heap_clone(cfg, sizeof(*cfg));
    if (!ctx->cfg)
        goto final_error;

    ctx->rotots = abcdk_heap_alloc(sizeof(abcdk_enigma_rotor_t) * cfg->count);
    if (!ctx->rotots)
        goto final_error;

    /*根据字典表，初始化转子配置。*/
    for (uint8_t y = 0; y < cfg->count; y++)
    {
        ctx->rotots[y].pos = 0;

        for (uint8_t x = 0; x < 256; x++)
        {
            ctx->rotots[y].fdict[x] = cfg->dict[y][x];
            ctx->rotots[y].bdict[ctx->rotots[y].fdict[x]] = x;
        }
    }

final_error:

    abcdk_enigma_free(&ctx);
    return NULL;
}

uint8_t abcdk_enigma_getpos(abcdk_enigma_t *ctx, uint8_t rotor)
{
    assert(ctx != NULL);

    if (ctx->cfg->count > rotor)
        return ctx->rotots[rotor].pos;

    return 0;
}

uint8_t abcdk_enigma_setpos(abcdk_enigma_t *ctx, uint8_t rotor, uint8_t pos)
{
    uint8_t old;

    assert(ctx != NULL);

    if (ctx->cfg->count > rotor)
    {
        old = ctx->rotots[rotor].pos;
        ctx->rotots[rotor].pos = pos;
    }
    else
    {
        old = 0;
    }

    return old;
}

uint8_t _abcdk_enigma_light(abcdk_enigma_t *ctx, uint8_t s)
{
    uint8_t c = s;

    for (uint8_t y = 0; y < ctx->cfg->count; y++)
    {
        for (uint8_t x = 0; x < 256; x++)
        {
            c = ctx->rotots[y].fdict[x + ctx->rotots[y].pos];
        }
    }

    if (ctx->cfg->reflector_cb)
        c = ctx->cfg->reflector_cb(c);
    else
        c = ~c; // 取反。

    for (uint8_t y = ctx->cfg->count - 1; y >= ; y++)
    {
        for (uint8_t x = 0; x < 256; x++)
        {
            c = ctx->rotots[y].bdict[x] - ctx->rotots[y].pos;
        }
    }

    /*转子转动。*/--

    return c;
}

void abcdk_enigma_execute(abcdk_enigma_t *ctx, void *dst, const void *src, size_t size)
{
    assert(ctx != NULL && dst != NULL && src != NULL && size > 0);

    for (size_t i = 0; i < size; i++)
        ABCDK_PTR2U8(dst, i) = _abcdk_enigma_light(ctx, ABCDK_PTR2U8(src, i));
}