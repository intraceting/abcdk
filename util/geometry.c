/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
*/
#include "abcdk-util/geometry.h"

double abcdk_line_length_3d(const abcdk_point_t *p1, const abcdk_point_t *p2)
{
    assert(p1 && p2);

    return fabs(sqrt(pow(p1->x - p2->x, 2) + pow(p1->y - p2->y, 2) + pow(p1->z - p2->z, 2)));
}

double abcdk_line_radian_2d(const abcdk_point_t *p1, const abcdk_point_t *p2, int axis)
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

void abcdk_point_shift_2d(const abcdk_point_t *p1, double radian, double dist, abcdk_point_t *p2)
{
    assert(p1 && p2);

    p2->x = p1->x + dist * cos(radian);
    p2->y = p1->y + dist * sin(radian);
}

void abcdk_resize_ratio_2d(abcdk_resize_t *ratio,double src_w, double src_h,double dst_w, double dst_h,int keep_ratio)
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

double abcdk_resize_src2dst_2d(const abcdk_resize_t *ratio,double src, int x)
{
    assert(ratio != NULL);
    assert(src >= 0.0);

    if (x)
        return (src * ratio->x_factor) + ratio->x_shift;

    return (src * ratio->y_factor) + ratio->y_shift;
}

double abcdk_resize_dst2src_2d(const abcdk_resize_t *ratio,double dst, int x)
{
    assert(ratio != NULL);
    assert(dst >= 0.0);

    if (x)
        return (dst - ratio->x_shift) / ratio->x_factor;

    return (dst - ratio->y_shift) / ratio->y_factor;
}

int abcdk_point_in_polygon_2d(const abcdk_point_t *p,const abcdk_polygon_t *polygon)
{
    abcdk_point_t p1,p2;
    int cross = 0;
    double x;
    int chk;

	for (size_t i = 0; i < polygon->numbers; i++)   
	{  
        /*点P1与P2形成连线段。*/
		p1 = polygon->points[i];  
		p2 = polygon->points[(i + 1) % polygon->numbers];
 
		if ( p1.y == p2.y )  
			continue;  
		if ( p->y < ABCDK_MIN(p1.y, p2.y) )  
			continue;  
		if ( p->y >= ABCDK_MAX(p1.y, p2.y) )  
			continue;  

		/*求交点的x坐标(由直线两点式方程转化而来)。*/
 
		x = (double)(p->y - p1.y) * (double)(p2.x - p1.x) / (double)(p2.y - p1.y) + p1.x;  
 
		/*只统计p1p2与p向右射线的交点。*/
		if ( x > p->x )  
			cross++;
 
	}  
 
	/* 
     * 交点为偶数，点在多边形之外。  
	 * 交点为奇数，点在多边形之内。 
     * 
     */
	chk = ((cross % 2) == 1);

	return chk;
}