/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/enigma/enigma.h"


/** Enigma转子。 */
typedef struct _abcdk_enigma_rotor
{
    /** 正向字典。*/
    uint8_t fdict[256];

    /** 逆向字典。*/
    uint8_t bdict[256];

    /** 步进指针。*/
    size_t pos;

} abcdk_enigma_rotor_t;

/** Enigma加密机。 */
struct _abcdk_enigma
{
    /** 转子数组。*/
    abcdk_enigma_rotor_t *rotors;

    /** 反射板字典。*/
    uint8_t rdict[256];

    /** 转子数量。*/
    size_t rows;

    /** 通道数量。*/
    size_t cols;

}; // abcdk_enigma_t;

void abcdk_enigma_mkdict(uint64_t *seed, uint8_t *dict, size_t rows, size_t cols)
{
    uint8_t c;
    int chk;

    assert(seed != NULL && dict != NULL && rows > 0);

    for (size_t y = 0; y < rows; y++)
    {
        /*生成连续不重复数字做为字典表。*/
        for (size_t x = 0; x < cols; x++)
            dict[y * cols + x] = x;

        /*使用洗牌算法打乱字典表顺序。*/
        abcdk_rand_shuffle_array(&dict[y * cols],cols,seed,1);
    }
}

void abcdk_enigma_free(abcdk_enigma_t **ctx)
{
    abcdk_enigma_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_heap_freep((void **)&ctx_p->rotors);

    abcdk_heap_free(ctx_p);
}

abcdk_enigma_t *abcdk_enigma_create(const uint8_t *dict,size_t rows, size_t cols)
{
    abcdk_enigma_t *ctx = NULL;
    abcdk_enigma_rotor_t *rotor = NULL;
    uint8_t chk_dict[256/8];
    uint8_t c;
    int chk;

    assert(dict != NULL && rows > 1 && cols >= 2 && cols <= 256 && cols % 2 == 0);

    /*检查字典表，每张字典表中的字符不能出现重复的。*/
    for (size_t y = 0; y < rows; y++)
    {
        memset(chk_dict, 0, sizeof(chk_dict));
        for (size_t x = 0; x < cols; x++)
        {
            c = dict[y * cols + x];

            ABCDK_ASSERT(c < cols,"转子中通道的值超出范围。");

            chk = abcdk_bloom_mark(chk_dict, sizeof(chk_dict),c);

            ABCDK_ASSERT(chk == 0,"在同一个转子中通道的值不能出现重复。");
        }
    }

    ctx = abcdk_heap_alloc(sizeof(abcdk_enigma_t));
    if (!ctx)
        return NULL;

    ctx->rotors = abcdk_heap_alloc(sizeof(abcdk_enigma_rotor_t) * rows);
    if (!ctx->rotors)
        goto ERR;

    /*根据字典表，初始化转子配置。*/
    for (size_t y = 0; y < rows - 1; y++)
    {
        ctx->rotors[y].pos = 0;

        for (size_t x = 0; x < cols; x++)
        {
            c = dict[y * cols + x];

            /*正向字典。*/
            ctx->rotors[y].fdict[x] = c;

            /*逆向字典（正向字典的索引）。*/
            ctx->rotors[y].bdict[c] = x;
        }
    }

    /*初始化反射板，形成对称映射。*/
    for (size_t x = 0; x < cols; x += 2)
    {
        uint8_t a = dict[(rows - 1) * cols + x];
        uint8_t b = dict[(rows - 1) * cols + x + 1];

        /*a和b互相映射。*/
        ctx->rdict[a] = b;
        ctx->rdict[b] = a;
    }

    /*验证反射板。*/
    for (int x = 0; x < cols; x++) 
    {
        assert(ctx->rdict[ctx->rdict[x]] == x);
    } 

    ctx->cols = cols;
    ctx->rows = rows-1;//最后一个轮子是反射板。

    return ctx;

ERR:

    abcdk_enigma_free(&ctx);
    return NULL;
}

abcdk_enigma_t *abcdk_enigma_create2(uint64_t seed,size_t rows, size_t cols)
{
    uint8_t *dict;
    abcdk_enigma_t *ctx;

    assert(rows > 0 && cols >= 2 && cols <= 256 && cols % 2 == 0);

    dict = (uint8_t*)abcdk_heap_alloc(rows * cols);
    if(!dict)
        return NULL;

    abcdk_enigma_mkdict(&seed,dict,rows,cols);

    ctx = abcdk_enigma_create(dict,rows,cols);
    abcdk_heap_free(dict);
    
    return ctx;
}

abcdk_enigma_t *abcdk_enigma_create3(uint64_t seed[],size_t rows,size_t cols)
{
    uint8_t *dict;
    abcdk_enigma_t *ctx;

    assert(seed != NULL && rows > 0 && cols >= 2 && cols <= 256 && cols % 2 == 0);

    dict = (uint8_t*)abcdk_heap_alloc(rows * cols);
    if(!dict)
        return NULL;

    for (size_t i = 0; i < rows; i++)
        abcdk_enigma_mkdict(&seed[i], &dict[i * cols], 1,cols);

    ctx = abcdk_enigma_create(dict,rows,cols);
    abcdk_heap_free(dict);
    
    return ctx;  
}

uint8_t abcdk_enigma_getpos(abcdk_enigma_t *ctx, size_t row)
{
    assert(ctx != NULL);
    assert(ctx->rows > row);

    return ctx->rotors[row].pos;
}

uint8_t abcdk_enigma_setpos(abcdk_enigma_t *ctx, size_t row, uint8_t pos)
{
    uint8_t old;

    assert(ctx != NULL);
    assert(ctx->rows > row && pos < ctx->cols);

    old = ctx->rotors[row].pos;
    ctx->rotors[row].pos = pos;

    return old;
}

static inline uint8_t _abcdk_enigma_light_v1(abcdk_enigma_t *ctx, uint8_t c)
{
    /*
     * 第一个转子转动一位。
     *
     * 1：如果第一个转子产生进位，那么POS变成0，第二个转子转动一位。
     * 2：如果第二个转子产生进位，那么POS变成0，第三个转子转动一位。
     * 3：……
     * 
    */
    for (size_t y = 0; y < ctx->rows; y++)
    {
        ctx->rotors[y].pos = (ctx->rotors[y].pos + 1) % ctx->cols;

        /*当POS变成0时，表示产生进位。*/
        if (ctx->rotors[y].pos)
            break;
    }

    /*
     * 正向通过转子。
     *
     * 1：转子转动的实质是rotor转动到POS的位置，那么实际上就是:
     * c = rotor->fdict[c + POS] 
     * 
     * 2：因为rotor只有COLS个元素，为了让转子形成环形转动效果，所以应该写成如下形式:
     * c = rotor->fdict[(c + POS)%COLS]
     * 
    */
    for (size_t y = 0; y < ctx->rows; y++)
    {
        c = ctx->rotors[y].fdict[(c + ctx->rotors[y].pos) % ctx->cols];
    }

    /* 通过反射板。*/
    c = ctx->rdict[c];

    /*
     * 逆向通过转子。
     * 
     * 1：转子转动的实质是rotor表在POS位置，那么实际上就是：
     * c = rotor->bdict[c] + (COLS- POS)
     * 
     * 2：因为rotor只有COLS个元素，为了让转子形成环形转动效果，所以应该写成如下形式:
     * c = (rotor->bdict[c] + (COLS- POS)) % COLS
     * 
    */
    for (size_t y = 0; y < ctx->rows; y++)
    {
        c = (ctx->rotors[ctx->rows - 1 - y].bdict[c] + (ctx->cols - ctx->rotors[ctx->rows - 1 - y].pos)) % ctx->cols;
    }

    return c;
}

static inline uint8_t _abcdk_enigma_light_v2(abcdk_enigma_t *ctx, uint8_t c)
{
    for (size_t y = 0; y < ctx->rows; y++)
    {
        abcdk_enigma_rotor_t *r = &ctx->rotors[y];

        r->pos = (r->pos + 1) % ctx->cols;

        if (r->pos)
            break;
    }

    for (size_t y = 0; y < ctx->rows; y++)
    {
        abcdk_enigma_rotor_t *r = &ctx->rotors[y];

        c = r->fdict[(c + r->pos) % ctx->cols];
    }

    c = ctx->rdict[c];

    for (size_t y = 0; y < ctx->rows; y++)
    {
        abcdk_enigma_rotor_t *r = &ctx->rotors[ctx->rows - 1 - y];

        c = (r->bdict[c] + (ctx->cols - r->pos)) % ctx->cols;
    }

    return c;
}

static inline uint8_t _abcdk_enigma_light_v3(abcdk_enigma_t *ctx, uint8_t c)
{
    for (size_t y = 0; y < ctx->rows; y++)
    {
        abcdk_enigma_rotor_t *r = &ctx->rotors[y];

        r->pos = (r->pos + 1) & (ctx->cols-1);//2的N次方才能用这种方法。

        if (r->pos)
            break;
    }

    for (size_t y = 0; y < ctx->rows; y++)
    {
        abcdk_enigma_rotor_t *r = &ctx->rotors[y];

        c = r->fdict[(c + r->pos) & (ctx->cols-1)];//2的N次方才能用这种方法。
    }

    c = ctx->rdict[c];

    for (size_t y = 0; y < ctx->rows; y++)
    {
        abcdk_enigma_rotor_t *r = &ctx->rotors[ctx->rows - 1 - y];

        c = (r->bdict[c] + (ctx->cols - r->pos)) & (ctx->cols-1);//2的N次方才能用这种方法。
    }

    return c;
}

static inline int _abcek_enigma_check_2N(size_t n)
{
    return (n > 0) && ((n & (n - 1)) == 0);//当n与n-1互为反码时，此数值为2的N次方。
}

uint8_t abcdk_enigma_light(abcdk_enigma_t *ctx, uint8_t c)
{
    assert(ctx != NULL);
    assert(c < ctx->cols);

    /*当通道数量为2的N次方时，可以启用加速代码。*/
    if(_abcek_enigma_check_2N(ctx->cols))
        return _abcdk_enigma_light_v3(ctx,c);
    
    return _abcdk_enigma_light_v2(ctx,c);
}

void abcdk_enigma_light_batch(abcdk_enigma_t *ctx,uint8_t *dst,const uint8_t *src,size_t size)
{
    const uint8_t *src_p;
    uint8_t *dst_p;
    assert(ctx != NULL && dst != NULL && src != NULL && size > 0);

    for (size_t i = 0; i < size; i++)
        dst[i] = abcdk_enigma_light(ctx, src[i]);
}

