/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_GEOMETRY_H
#define ABCDK_UTIL_GEOMETRY_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

/** 空间点坐标。*/
typedef struct _abcdk_point
{
    /** X轴坐标。 */
    double x;

    /** Y轴坐标。 */
    double y;

    /** Z轴坐标。 */
    double z;
} abcdk_point_t;

/** 尺寸变换系数。*/
typedef struct _abcdk_resize_scale
{
	/** X轴因子。 */
	double x_factor;

    /** Y轴因子。 */
	double y_factor;

    /** X轴位移。*/
	double x_shift;

    /** Y轴位移。*/
	double y_shift;

}abcdk_resize_scale_t;

/**
 * 计算空间两点之间的直线距离。
 * 
 * @param b 起点。
 * @param e 终点。
 */
double abcdk_line_length_3d(const abcdk_point_t *b, const abcdk_point_t *e);

/**
 * 计算平面中的射线弧度。
 * 
 * @note 角度=弧度*180/PI(3.1415926...)。
 * 
 * @param b 起点。
 * @param e 终点。
 * @param axis 仅支持X轴或者Y轴。
 * 
*/
double abcdk_line_radian_2d(const abcdk_point_t *b, const abcdk_point_t *e, int axis);

/**
 * 计算平面中点的位移坐标。
 * 
 * @param b 起点。
 * @param radian 弧度。
 * @param dist 距离。
 * @param e 终点。
 */
void abcdk_point_shift_2d(const abcdk_point_t *b,double radian,double dist,abcdk_point_t *e);

/**
 * 生成尺寸变换系数。
 * 
 * @param keep_ratio !0 保持纵横比，0 不保持纵横比。
*/
void abcdk_resize_ratio_2d(abcdk_resize_scale_t *ratio,double src_w, double src_h,double dst_w, double dst_h,int keep_ratio);

/**
 * 源图到目标图坐标变换。
 * 
 * @param x !0 X轴坐标，0 Y轴坐标。
*/
double abcdk_resize_src2dst_2d(const abcdk_resize_scale_t *ratio,double src, int x_or_y);

/**
 * 目标图到源图坐标变换。
 * 
 * @param x !0 X轴坐标，0 Y轴坐标。
*/
double abcdk_resize_dst2src_2d(const abcdk_resize_scale_t *ratio,double dst, int x_or_y);

/**
 * 测试点是否多边形内部。
 * 
 * @param polygon 多边形顶点坐标，顺时针排列。
 * @param numbers 顶点数量。
 * 
 * @return !0 在内部，0 在外部。
*/
int abcdk_point_in_polygon_2d(const abcdk_point_t *p,const abcdk_point_t *polygon,size_t numbers);

/**
 * 计算两条直接交点。
 *
 * @return 0 交点在延长线上，1 交点在第一条线段，2 交点在第二条线段上，3 交点在两条线段上，-1 无交点(平行或共线)。
*/
int abcdk_line_cross_2d(const abcdk_point_t *line1_b, const abcdk_point_t *line1_e,
                        const abcdk_point_t *line2_b, const abcdk_point_t *line2_e,
                        abcdk_point_t *p);



__END_DECLS

#endif //ABCDK_UTIL_GEOMETRY_H