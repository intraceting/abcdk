/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/geometry.h"

double abcdk_line_length_3d(const abcdk_point_t *b, const abcdk_point_t *e)
{
    assert(b && e);

    return fabs(sqrt(pow(b->x - e->x, 2) + pow(b->y - e->y, 2) + pow(b->z - e->z, 2)));
}

double abcdk_line_radian_2d(const abcdk_point_t *b, const abcdk_point_t *e, int axis)
{
    double radian;

    assert(b && e);

    /*To upper*/
    axis = toupper(axis);
    assert((axis == 'X') || (axis == 'Y'));

    if (axis == 'X')
        radian = atan2((e->y - b->y), (e->x - b->x));
    else if (axis == 'Y')
        radian = atan2((e->x - b->x), (e->y - b->y));

    return radian;
}

void abcdk_point_shift_2d(const abcdk_point_t *b, double radian, double dist, abcdk_point_t *e)
{
    assert(b && e);

    e->x = b->x + dist * cos(radian);
    e->y = b->y + dist * sin(radian);
}

void abcdk_resize_ratio_2d(abcdk_resize_scale_t *ratio, double src_w, double src_h, double dst_w, double dst_h, int keep_ratio)
{
    double min_factor;

    assert(ratio != NULL);
    assert(src_w > 0 && src_h > 0);
    assert(dst_w > 0 && dst_h > 0);

    ratio->x_factor = dst_w / src_w;
    ratio->y_factor = dst_h / src_h;

    if (keep_ratio)
    {
        min_factor = ABCDK_MIN(ratio->x_factor, ratio->y_factor);
        ratio->x_factor = min_factor;
        ratio->y_factor = min_factor;
    }

    ratio->x_shift = (dst_w - (ratio->x_factor * src_w)) / 2.0;
    ratio->y_shift = (dst_h - (ratio->y_factor * src_h)) / 2.0;
}

double abcdk_resize_src2dst_2d(const abcdk_resize_scale_t *ratio, double src, int x_or_y)
{
    assert(ratio != NULL);
    assert(src >= 0.0);

    if (x_or_y)
        return (src * ratio->x_factor) + ratio->x_shift;

    return (src * ratio->y_factor) + ratio->y_shift;
}

double abcdk_resize_dst2src_2d(const abcdk_resize_scale_t *ratio, double dst, int x_or_y)
{
    assert(ratio != NULL);
    assert(dst >= 0.0);

    if (x_or_y)
        return (dst - ratio->x_shift) / ratio->x_factor;

    return (dst - ratio->y_shift) / ratio->y_factor;
}

int abcdk_point_in_polygon_2d(const abcdk_point_t *p, const abcdk_point_t *polygon, size_t numbers)
{
    abcdk_point_t b, e;
    int cross = 0;
    double x;
    int chk;

    assert(p != NULL && polygon != NULL);

    for (size_t i = 0; i < numbers; i++)
    {
        /*点b与e形成连线段.*/
        b = polygon[i];
        e = polygon[(i + 1) % numbers]; // 最后的点连起来, 组成封闭的多边形.

        if (b.y == e.y)
            continue;
        if (p->y < ABCDK_MIN(b.y, e.y))
            continue;
        if (p->y >= ABCDK_MAX(b.y, e.y))
            continue;

        /*求交点的x坐标(由直线两点式方程转化而来).*/
        x = (double)(p->y - b.y) * (double)(e.x - b.x) / (double)(e.y - b.y) + b.x;

        /*只统计b和e与p向右射线的交点.*/
        if (x > p->x)
            cross++;
    }

    /*
     * 交点为偶数, 点在多边形之外.
     * 交点为奇数, 点在多边形之内.
     *
     */
    chk = ((cross % 2) == 1);

    return chk;
}

int abcdk_line_cross_2d(const abcdk_point_t *line1_b, const abcdk_point_t *line1_e,
                        const abcdk_point_t *line2_b, const abcdk_point_t *line2_e,
                        abcdk_point_t *p)
{
    double a1, b1, c1;
    double a2, b2, c2;
    double d;
    double rx0, ry0, rx1, ry1;
    int chk1, chk2;

    assert(line1_b != NULL && line1_e != NULL);
    assert(line2_b != NULL && line2_e != NULL);
    assert(p != NULL);

    a1 = line1_e->y - line1_b->y;
    b1 = line1_b->x - line1_e->x;
    c1 = a1 * line1_b->x + b1 * line1_b->y;
    a2 = line2_e->y - line2_b->y;
    b2 = line2_b->x - line2_e->x;
    c2 = a2 * line2_b->x + b2 * line2_b->y;
    d = a1 * b2 - a2 * b1;

    /*如果分母为0, 则平行或共线, 无交点.*/
    if (d == 0)
        return -1;

    p->x = (b2 * c1 - b1 * c2) / d;
    p->y = (a1 * c2 - a2 * c1) / d;

    rx0 = (p->x - line1_b->x) / (line1_e->x - line1_b->x),
    ry0 = (p->y - line1_b->y) / (line1_e->y - line1_b->y),
    rx1 = (p->x - line2_b->x) / (line2_e->x - line2_b->x),
    ry1 = (p->y - line2_b->y) / (line2_e->y - line2_b->y);

    /* 判断交点是否在线段1上.*/
    chk1 = ((rx0 >= 0 && rx0 <= 1) || (ry0 >= 0 && ry0 <= 1));
    /* 判断交点是否在线段2上.*/
    chk2 = ((rx1 >= 0 && rx1 <= 1) || (ry1 >= 0 && ry1 <= 1));

    if (chk1 && chk2)
        return 3;
    else if (chk2)
        return 2;
    else if (chk1)
        return 1;

    return 0;
}
