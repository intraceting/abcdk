/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "geometry.h"

double abcdk_line_segment_length(const abcdk_point_t *p1, const abcdk_point_t *p2)
{
    assert(p1 && p2);

    return fabs(sqrt(pow(p1->x - p2->x, 2) + pow(p1->y - p2->y, 2) + pow(p1->z - p2->z, 2)));
}

double abcdk_half_line_radian(const abcdk_point_t *p1, const abcdk_point_t *p2, int axis)
{
    double radian;

    assert(p1 && p2);

    /*To upper*/
    axis = toupper(axis);
    assert((axis == 'X') || (axis == 'Y'));

    if (axis == 'X')
        radian = atan2((p2->y - p1->y), (p2->x - p1->x));
    else if (axis == 'Y')
        radian = atan2((p2->x - p1->x), (p2->y - p1->y));

    return radian;
}

void abcdk_point_shift(const abcdk_point_t *p1, double radian, double dist, abcdk_point_t *p2)
{
    assert(p1 && p2);

    p2->x = p1->x + dist * cos(radian);
    p2->y = p1->y + dist * sin(radian);
}

void abcdk_resize_make(abcdk_resize_t *ratio,
                       double src_w, double src_h,
                       double dst_w, double dst_h,
                       int keep_ratio)
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

double abcdk_resize_src2dst(const abcdk_resize_t *ratio,
                            double src, int x)
{
    assert(ratio != NULL);
    assert(src >= 0.0);

    if (x)
        return (src * ratio->x_factor) + ratio->x_shift;

    return (src * ratio->y_factor) + ratio->y_shift;
}

double abcdk_resize_dst2src(const abcdk_resize_t *ratio,
                            double dst, int x)
{
    assert(ratio != NULL);
    assert(dst >= 0.0);

    if (x)
        return (dst - ratio->x_shift) / ratio->x_factor;

    return (dst - ratio->y_shift) / ratio->y_factor;
}