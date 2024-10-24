/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/register.h"

/** 简单的寄存器。*/
typedef union _abcdk_register
{
    volatile uint8_t u8;
    volatile uint16_t u16;
    volatile uint32_t u32;
    volatile uint64_t u64;
}abcdk_register_t;

volatile void *abcdk_register(int type, uint8_t addr)
{
    static volatile abcdk_register_t reg[256] = {0};
    volatile void *p = NULL;

    ABCDK_ASSERT((type == 8 || type == 16 || type == 32 || type == 64),"仅支持8位、16位、32位、64位四种类型。");

    switch (type)
    {
    case 8:
        p = &reg[addr].u8;
    case 16:
        p = &reg[addr].u16;
    case 32:
        p = &reg[addr].u32;
    case 64:
    default:
        p = &reg[addr].u64;
    }
    
    return p;
}