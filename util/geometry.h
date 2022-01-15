/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_GEOMETRY_H
#define ABCDK_UTIL_GEOMETRY_H

#include "util/general.h"

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
typedef struct _abcdk_resize
{
	/** X轴因子。 */
	double x_factor;

    /** Y轴因子。 */
	double y_factor;

    /** X轴位移。*/
	double x_shift;

    /** Y轴位移。*/
	double y_shift;

}abcdk_resize_t;

/** 多边型。*/
typedef struct _abcdk_polygon
{
    /** 顶点数量。*/
    size_t numbers;
    
    /** 顶点坐标。*/
    abcdk_point_t *points;

}abcdk_polygon_t;


/**
 * 计算空间两点之间的直线距离。
 * 
 * @param p1 起点。
 * @param p2 终点。
 */
double abcdk_line_length_3d(const abcdk_point_t *p1, const abcdk_point_t *p2);

/**
 * 计算平面中的射线弧度。
 * 
 * Z轴忽略。
 * 
 * 角度=弧度*180/PI(3.1415926...)。
 * 
 * @param p1 起点。
 * @param p2 终点。
 * @param axis 仅支持X轴或者Y轴。
 * 
*/
double abcdk_line_radian_2d(const abcdk_point_t *p1, const abcdk_point_t *p2, int axis);

/**
 * 计算平面中点的位移坐标。
 * 
 * Z轴忽略。
 * 
 * @param p1 起点。
 * @param radian 弧度。
 * @param dist 距离。
 * @param p2 终点。
 */
void abcdk_point_shift_2d(const abcdk_point_t *p1,double radian,double dist,abcdk_point_t *p2);

/**
 * 生成尺寸变换系数。
 * 
 * @param keep_ratio !0 保持纵横比，0 不保持纵横比。
*/
void abcdk_resize_ratio_2d(abcdk_resize_t *ratio,double src_w, double src_h,double dst_w, double dst_h,int keep_ratio);

/**
 * 源图到目标图坐标变换。
 * 
 * @param x !0 X轴坐标，0 Y轴坐标。
*/
double abcdk_resize_src2dst_2d(const abcdk_resize_t *ratio,double src, int x);

/**
 * 目标图到源图坐标变换。
 * 
 * @param x !0 X轴坐标，0 Y轴坐标。
*/
double abcdk_resize_dst2src_2d(const abcdk_resize_t *ratio,double dst, int x);

/**
 * 测试点是否多边形内部。
 * 
 * @return !0 在内部，0 在外部。
*/
int abcdk_point_in_polygon_2d(const abcdk_point_t *p,const abcdk_polygon_t *polygon);

__END_DECLS

#endif //ABCDK_UTIL_GEOMETRY_H