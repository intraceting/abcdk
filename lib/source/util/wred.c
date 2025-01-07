/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/wred.h"

/**简单的WRED(加权随机早期检测)算法。 */
struct _abcdk_wred
{
    /**最小阈值。 */
    int min_th;

    /**最大阈值。 */
    int max_th;

    /**权重因子。 */
    double weight;

    /**概率因子。 */
    double prob;

    /**均值。*/    
    double avg;
};//abcdk_wred_t;

static double _abcdk_wred_update_avg(double avg, int qlen, double weight)
{
    return (1 - weight) * avg + weight * qlen;
}

static double _abcdk_wred_update_drop_prob(double avg, int min_th, int max_th, double max_p)
{
    if (avg < min_th)
        return 0.0; // 队列平均长度小于min_th时，最小概率。
    else if (avg > max_th)
        return 1.0; // 队列平均长度大于max_th时，最大概率。
    else
        return (double)((avg - min_th) / (max_th - min_th)) * max_p; // 在min_th和max_th之间时，均值概率。
}

void abcdk_wred_destroy(abcdk_wred_t **ctx)
{
    abcdk_wred_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_heap_free(ctx_p);
}

abcdk_wred_t *abcdk_wred_create(int min_th,int max_th,int weight,int prob)
{
    abcdk_wred_t *ctx;

    assert(min_th > 0 && max_th > 0 && weight > 0 && prob > 0);
    assert(min_th < max_th);
    assert(weight >= 1 && weight <= 99);
    assert(prob >= 1 && prob <= 99);

    ctx = (abcdk_wred_t*)abcdk_heap_alloc(sizeof(abcdk_wred_t));
    if(!ctx)
        return NULL;

    ctx->min_th = min_th;
    ctx->max_th = max_th;
    ctx->weight = (double)weight/100.;
    ctx->prob = (double)prob/100.;
    ctx->avg = 0.0;

    return ctx;
}

int abcdk_wred_update(abcdk_wred_t *ctx,int qlen)
{
    double drop_prob,drop_rand;

    assert(ctx != NULL && qlen >= 0);

    /*计算均值。*/
    ctx->avg = _abcdk_wred_update_avg(ctx->avg,qlen,ctx->weight);

    /*计算丢包概率。*/
    drop_prob = _abcdk_wred_update_drop_prob(ctx->avg,ctx->min_th,ctx->max_th,ctx->prob);

    /*获取随机概率。*/
    drop_rand = (double)rand() / RAND_MAX; 

    /*当随机概率小于丢包概率时丢弃，反之则保留。*/
    if(drop_rand < drop_prob)
        return 1;

    return 0;
}