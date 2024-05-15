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

char *abcdk_rand_string(char *buf, size_t size, int type)
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
    assert(type >= 0 && type <= 4);

    for (int i = 0; i < size; i++)
    {
        uint64_t rand_num = abcdk_rand_number();

        if (0 == type)
            buf[i] = dict_printable[rand_num % sizeof(dict_printable)];
        else if( 1 == type)
            buf[i] = dict_alnum[rand_num % sizeof(dict_alnum)];
        else if( 2 == type)
            buf[i] = dict_uppercase[rand_num % sizeof(dict_uppercase)];
        else if( 3 == type)
            buf[i] = dict_lowercase[rand_num % sizeof(dict_lowercase)];
        else if( 4 == type)
            buf[i] = dict_digit[rand_num % sizeof(dict_digit)];
    }

    return buf;
}
