/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/random.h"

int64_t abcdk_rand(uint64_t *seed)
{
    uint64_t next;

    assert(seed != NULL);

    if(*seed == 0)
        *seed = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,6);

    next = *seed;

    /*POSIX.1-2001*/

    next = next * 1103515245UL + 12345UL;
    next = next / 65536;

    /*copy.*/
    *seed = next;

    return next;
}

int64_t abcdk_rand_q()
{
    static uint64_t seed = 0;
    int64_t num;

    abcdk_atomic_lock();
    num = abcdk_rand(&seed);
    abcdk_atomic_unlock();

    return num;
}

char *abcdk_rand_bytes(char *buf, size_t size, int type)
{
    static char dict_printable[] = {
        ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
        '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
        '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
        'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'};

    static char dict_alnum[] = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

    static char dict_uppercase[] = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};

    static char dict_lowercase[] = {
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

    static char dict_digit[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

    assert(buf != NULL && size > 0);
    assert(type >= 0 && type <= 5);

    for (int i = 0; i < size; i++)
    {
        uint64_t rand_num = abcdk_rand_number();

        if (0 == type)
            buf[i] = dict_printable[rand_num % sizeof(dict_printable)];
        else if (1 == type)
            buf[i] = dict_alnum[rand_num % sizeof(dict_alnum)];
        else if (2 == type)
            buf[i] = dict_uppercase[rand_num % sizeof(dict_uppercase)];
        else if (3 == type)
            buf[i] = dict_lowercase[rand_num % sizeof(dict_lowercase)];
        else if (4 == type)
            buf[i] = dict_digit[rand_num % sizeof(dict_digit)];
        else if (5 == type)
            buf[i] = (int8_t)(rand_num % 256);
    }

    return buf;
}

void abcdk_rand_shuffle(uint64_t *seed, size_t size, abcdk_rand_shuffle_swap_cb swap_cb, void *opaque)
{
    assert(seed != NULL && size > 0 && swap_cb != NULL);

    /*洗牌算法(Fisher-Yates)打乱顺序。*/
    for (size_t a = size - 1; a > 0; a--)
    {
        /*生成一个0到a的随机整数。*/
        size_t b = (uint64_t)abcdk_rand(seed) % (a + 1);

        /*交换a和b。*/
        swap_cb(a, b, opaque);
    }
}

static void _abcdk_rand_shuffle_array_swap_uint8_cb(size_t a,size_t b, void *opaque)
{
    uint8_t *array_p = (uint8_t*)opaque;

    ABCDK_INTEGER_SWAP(array_p[a],array_p[b]);
}

static void _abcdk_rand_shuffle_array_swap_uint16_cb(size_t a,size_t b, void *opaque)
{
    uint16_t *array_p = (uint16_t*)opaque;

    ABCDK_INTEGER_SWAP(array_p[a],array_p[b]);
}

static void _abcdk_rand_shuffle_array_swap_uint32_cb(size_t a,size_t b, void *opaque)
{
    uint32_t *array_p = (uint32_t*)opaque;

    ABCDK_INTEGER_SWAP(array_p[a],array_p[b]);
}

static void _abcdk_rand_shuffle_array_swap_uint64_cb(size_t a,size_t b, void *opaque)
{
    uint64_t *array_p = (uint64_t*)opaque;

    ABCDK_INTEGER_SWAP(array_p[a],array_p[b]);
}

static void _abcdk_rand_shuffle_array_swap_float_cb(size_t a,size_t b, void *opaque)
{
    float *array_p = (float*)opaque;

    float tmp = array_p[a];
    array_p[a] = array_p[b];
    array_p[b] = tmp;
}

static void _abcdk_rand_shuffle_array_swap_double_cb(size_t a,size_t b, void *opaque)
{
    double *array_p = (double*)opaque;

    double tmp = array_p[a];
    array_p[a] = array_p[b];
    array_p[b] = tmp;
}

void *abcdk_rand_shuffle_array(void *buf,size_t count,uint64_t *seed,int type)
{
    assert(buf != NULL && count > 0 && seed != NULL && type >= 1 && type <= 6);

    if(type == 1)
        abcdk_rand_shuffle(seed,count,_abcdk_rand_shuffle_array_swap_uint8_cb,buf);
    else if(type == 2)
        abcdk_rand_shuffle(seed,count,_abcdk_rand_shuffle_array_swap_uint16_cb,buf);
    else if(type == 3)
        abcdk_rand_shuffle(seed,count,_abcdk_rand_shuffle_array_swap_uint32_cb,buf);
    else if(type == 4)
        abcdk_rand_shuffle(seed,count,_abcdk_rand_shuffle_array_swap_uint64_cb,buf);
    else if(type == 5)
        abcdk_rand_shuffle(seed,count,_abcdk_rand_shuffle_array_swap_float_cb,buf);
    else if(type == 6)
        abcdk_rand_shuffle(seed,count,_abcdk_rand_shuffle_array_swap_double_cb,buf);

    return buf;
}
