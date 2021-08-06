/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include "abcdk-util/geometry.h"


void test1()
{
    
    abcdk_point_t p1 = {0,0,0};

    abcdk_point_t p2= {95,55,0};

    double d = abcdk_geom_p2p_distance(&p1,&p2);

    printf("d=%lf\n",d);

    double r = abcdk_geom_halfline_radian(&p1,&p2,'X');
    double a = r*180/M_PIl;
    printf("r=%lf;a=%lf\n",r,a);

    abcdk_point_t p3 = {0,0,0};

    abcdk_geom_point_move(&p1,r,d/2,&p3);

    printf("x = %lf y = %lf\n",p3.x,p3.y);

    double d2 = abcdk_geom_p2p_distance(&p1,&p3);

    printf("d2=%lf\n",d2);
}

void test2()
{
    abcdk_point_t p = {0,0,0};

    //abcdk_point_t p1 = {99,95,0};
    //abcdk_point_t p2 = {90,90,0};

    abcdk_point_t p1 = {90,90,0};
    abcdk_point_t p2 = {81,85,0};

    double d1 = abcdk_geom_p2p_distance(&p,&p1);
    double d2 = abcdk_geom_p2p_distance(&p,&p2);
    double r1 = abcdk_geom_halfline_radian(&p,&p1,'X');
    double r2 = abcdk_geom_halfline_radian(&p,&p2,'X');
    double R1 = r1*180/M_PIl;
    double R2 = r2*180/M_PIl;

    printf("d1(%lf)-d2(%lf)=%lf\n",d1,d2,d1-d2);
    printf("r1(%lf)-r2(%lf)=%lf\n",r1,r2,r1-r2);
    printf("R1(%lf)-R2(%lf)=%lf\n",R1,R2,R1-R2);
}

void GuestLocation(const abcdk_point_t *p1,const abcdk_point_t *p2,double xspeed,abcdk_point_t *p3)
{
    double d = abcdk_geom_p2p_distance(p1,p2);
    double r = abcdk_geom_halfline_radian(p1,p2,'X');
    abcdk_geom_point_move(p2,r,d*xspeed,p3);
}

void test3()
{
    abcdk_point_t p1 = {45,45,0};
    abcdk_point_t p2 = {0,0,0};
    abcdk_point_t p3 = {0,0,0};

    GuestLocation(&p1,&p2,1,&p3);
    printf("p1={%lf,%lf},p2={%lf,%lf},p3={%lf,%lf}\n",p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);

    GuestLocation(&p1,&p2,2,&p3);
    printf("p1={%lf,%lf},p2={%lf,%lf},p3={%lf,%lf}\n",p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);

    GuestLocation(&p1,&p2,3,&p3);
    printf("p1={%lf,%lf},p2={%lf,%lf},p3={%lf,%lf}\n",p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);

    GuestLocation(&p1,&p2,0.1,&p3);
    printf("p1={%lf,%lf},p2={%lf,%lf},p3={%lf,%lf}\n",p1.x,p1.y,p2.x,p2.y,p3.x,p3.y);


}

int main(int argc, char **argv)
{
    //test1();
    //test2();

 //   test3();




    return 0;
}