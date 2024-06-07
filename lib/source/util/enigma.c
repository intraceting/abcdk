/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/enigma.h"

#define ABCDK_ENIGMA_ROW_MAX 32768
#define ABCDK_ENIGMA_COL_MAX 65536

/** Enigma转子。 */
typedef struct _abcdk_enigma_rotor
{
    /** 正向字典。*/
    uint16_t fdict[ABCDK_ENIGMA_COL_MAX];
    /** 逆向字典。*/
    uint16_t bdict[ABCDK_ENIGMA_COL_MAX];
    /** 步进指针。*/
    uint16_t pos;

} abcdk_enigma_rotor_t;

/** Enigma加密机。 */
struct _abcdk_enigma
{
    /** 转子数组。*/
    abcdk_enigma_rotor_t *rotors;

    /** 反射板字典。*/
    uint16_t rdict[ABCDK_ENIGMA_COL_MAX];

    /** 字典行数。*/
    size_t rows;

    /** 字典列数。*/
    size_t cols;

}; // abcdk_enigma_t;

void abcdk_enigma_mkdict(uint64_t *seed, uint16_t *dict, size_t rows, size_t cols)
{
    uint16_t c;
    int chk;

    assert(seed != NULL && dict != NULL && rows > 0 && rows <= ABCDK_ENIGMA_ROW_MAX && cols >= 4 && cols <= ABCDK_ENIGMA_COL_MAX && cols % 2 == 0);

    for (size_t y = 0; y < rows; y++)
    {
        /*生成连续不重复数字做为字典表。*/
        for (size_t x = 0; x < cols; x++)
            dict[y * cols + x] = x;

        /*使用洗牌算法(Fisher-Yates)打乱字典表顺序。*/
        for (size_t i = cols - 1; i > 0; i--)
        {
            /*生成一个0到i的随机整数。*/
            size_t j = (uint64_t)abcdk_rand(seed)%(i+1);

            /*交换dict[ROWS+j]和dict[ROWS+i]。*/
            ABCDK_INTEGER_SWAP(dict[y * cols + j],dict[y * cols + i]);
        }
    }
}

void abcdk_enigma_free(abcdk_enigma_t **ctx)
{
    abcdk_enigma_t *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_heap_free2((void **)&ctx_p->rotors);

    abcdk_heap_free(ctx_p);
}

abcdk_enigma_t *abcdk_enigma_create(const uint16_t *dict,size_t rows,size_t cols)
{
    abcdk_enigma_t *ctx = NULL;
    abcdk_enigma_rotor_t *rotor = NULL;
    uint8_t chk_dict[ABCDK_ENIGMA_COL_MAX/8];
    uint16_t c;
    int chk;

    assert(dict != NULL && rows >= 3 && rows <= ABCDK_ENIGMA_ROW_MAX && cols >= 4 && cols <= ABCDK_ENIGMA_COL_MAX && cols % 2 == 0);

    /*检查字典表，每张字典表中的字符不能出现重复的。*/
    for (size_t y = 0; y < rows; y++)
    {
        memset(chk_dict, 0, sizeof(chk_dict));
        for (size_t x = 0; x < cols; x++)
        {
            c = dict[y * cols + x];

            ABCDK_ASSERT(c < cols,"字典数据的值必须小于列宽度(转子的通道)。");

            chk = abcdk_bloom_mark(chk_dict, sizeof(chk_dict),c);

            ABCDK_ASSERT(chk == 0,"在同一行字典数据的值不能出现重复。");
        }
    }

    ctx = abcdk_heap_alloc(sizeof(abcdk_enigma_t));
    if (!ctx)
        return NULL;

    ctx->rotors = abcdk_heap_alloc(sizeof(abcdk_enigma_rotor_t) * rows);
    if (!ctx->rotors)
        goto final_error;

    /*根据字典表，初始化转子配置。*/
    for (size_t y = 0; y < rows; y++)
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

    /*初始化反射板。两个索引之间形成互查字典。*/
    for (size_t x = 0; x < cols; x++)
    {
        ctx->rdict[x] = cols - x -1;
    }
 

    ctx->cols = cols;
    ctx->rows = rows;

    return ctx;

final_error:

    abcdk_enigma_free(&ctx);
    return NULL;
}

abcdk_enigma_t *abcdk_enigma_create2(uint64_t seed,size_t rows,size_t cols)
{
    uint16_t *dict;
    abcdk_enigma_t *ctx;

    assert(rows > 0 && rows <= ABCDK_ENIGMA_ROW_MAX && cols >= 4 && cols <= ABCDK_ENIGMA_COL_MAX && cols % 2 == 0);

    dict = (uint16_t*)abcdk_heap_alloc(sizeof(uint16_t) * rows * cols);
    if(!dict)
        return NULL;

    abcdk_enigma_mkdict(&seed,dict,rows,cols);

    ctx = abcdk_enigma_create(dict,rows,cols);
    abcdk_heap_free(dict);
    
    return ctx;
}

uint16_t abcdk_enigma_getpos(abcdk_enigma_t *ctx, uint16_t rotor)
{
    assert(ctx != NULL);

    if (ctx->rows > rotor)
        return ctx->rotors[rotor].pos;

    return 0;
}

uint16_t abcdk_enigma_setpos(abcdk_enigma_t *ctx, uint16_t rotor, uint16_t pos)
{
    uint16_t old;

    assert(ctx != NULL);

    if (ctx->rows > rotor)
    {
        old = ctx->rotors[rotor].pos;
        ctx->rotors[rotor].pos = pos % ctx->cols;
    }
    else
    {
        old = 0;
    }

    return old;
}

uint16_t abcdk_enigma_light(abcdk_enigma_t *ctx, uint16_t s)
{
    uint16_t c;

    assert(ctx != NULL);
    assert(s < ctx->cols);

    c = s;

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

void abcdk_enigma_light_batch_u16(abcdk_enigma_t *ctx,uint16_t *dst,const uint16_t *src,size_t size)
{
    assert(ctx != NULL && dst != NULL && src != NULL && size > 0);

    for (size_t i = 0; i < size; i++)
        dst[i] = abcdk_enigma_light(ctx, src[i]);
}

void abcdk_enigma_light_batch_u8(abcdk_enigma_t *ctx,uint8_t *dst,const uint8_t *src,size_t size)
{
    assert(ctx != NULL && dst != NULL && src != NULL && size > 0);
    assert(ctx->cols <= 256);//单个轮子超过256个通道时，不能用使用此接口。

    for (size_t i = 0; i < size; i++)
        dst[i] = abcdk_enigma_light(ctx, src[i]);
}

