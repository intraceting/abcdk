/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/random.h"

static pid_t _abcdk_rand_globe_pid = -1;
static int _abcdk_rand_globe_fd = -1;

static void _abcdk_rand_globe_fd_close_cb(void)
{
    abcdk_closep(&_abcdk_rand_globe_fd);
}

static void _abcdk_rand_globe_fd_open(void)
{
    abcdk_atomic_lock();

    /*如果已经在子进程中，则关闭。*/
    if(_abcdk_rand_globe_pid != getpid())
        abcdk_closep(&_abcdk_rand_globe_fd);

    if (_abcdk_rand_globe_fd < 0)
    {
        _abcdk_rand_globe_fd = abcdk_open("/dev/urandom", 0, 0, 0);
        assert(_abcdk_rand_globe_fd >= 0);

        /*记录当前进程PID。*/
        _abcdk_rand_globe_pid = getpid();

        /*注册退出前关闭函数。*/
        atexit(_abcdk_rand_globe_fd_close_cb);
    }
    
    abcdk_atomic_unlock();
}

static ssize_t _abcdk_rand_read(void *buf, size_t size)
{
    ssize_t rlen = 0;

#if 0
    rlen = abcdk_load("/dev/urandom", buf, size, 0);
#else
    _abcdk_rand_globe_fd_open();
    rlen = read(_abcdk_rand_globe_fd,buf, size);
#endif

    return rlen;
}

uint64_t abcdk_rand(uint64_t min, uint64_t max)
{
    uint64_t num;
    int chk;

    assert(min < UINT64_MAX && max < UINT64_MAX);
    assert(min < max);

    chk = _abcdk_rand_read( &num, sizeof(num));
    if (chk != sizeof(num))
        num = rand();

    /*限制到区间内。*/
    num = num % (max - min + 1) + min;

    return num;
}

uint8_t *abcdk_rand_bytes(uint8_t *buf, size_t size, int type)
{
    int chk;

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

    chk = _abcdk_rand_read(buf, size);
    //assert(chk == size);

    for (int i = 0; i < size; i++)
    {
        if (0 == type)
            buf[i] = dict_printable[buf[i] % sizeof(dict_printable)];
        else if (1 == type)
            buf[i] = dict_alnum[buf[i] % sizeof(dict_alnum)];
        else if (2 == type)
            buf[i] = dict_uppercase[buf[i] % sizeof(dict_uppercase)];
        else if (3 == type)
            buf[i] = dict_lowercase[buf[i] % sizeof(dict_lowercase)];
        else if (4 == type)
            buf[i] = dict_digit[buf[i] % sizeof(dict_digit)];
        else if (5 == type)
            break;
    }

    return buf;
}

void abcdk_rand_shuffle(size_t size, abcdk_rand_shuffle_swap_cb swap_cb, void *opaque)
{
    assert(size > 0 && swap_cb != NULL);

    /*洗牌算法(Fisher-Yates)打乱顺序。*/
    for (size_t a = size - 1; a > 0; a--)
    {
        /*生成一个0到a的随机整数。*/
        size_t b = abcdk_rand(0,a);

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

void *abcdk_rand_shuffle_array(void *buf,size_t count,int type)
{
    assert(buf != NULL && count > 0 && type >= 1 && type <= 6);

    if(type == 1)
        abcdk_rand_shuffle(count,_abcdk_rand_shuffle_array_swap_uint8_cb,buf);
    else if(type == 2)
        abcdk_rand_shuffle(count,_abcdk_rand_shuffle_array_swap_uint16_cb,buf);
    else if(type == 3)
        abcdk_rand_shuffle(count,_abcdk_rand_shuffle_array_swap_uint32_cb,buf);
    else if(type == 4)
        abcdk_rand_shuffle(count,_abcdk_rand_shuffle_array_swap_uint64_cb,buf);
    else if(type == 5)
        abcdk_rand_shuffle(count,_abcdk_rand_shuffle_array_swap_float_cb,buf);
    else if(type == 6)
        abcdk_rand_shuffle(count,_abcdk_rand_shuffle_array_swap_double_cb,buf);

    return buf;
}
