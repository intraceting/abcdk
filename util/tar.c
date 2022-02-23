/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "util/tar.h"

int abcdk_tar_num2char(uintmax_t val, char *buf, size_t len)
{
    char *tmpbuf;
    size_t tmplen;
    uintmax_t tmpval;

    assert(buf != NULL && len > 0);

    tmpbuf = buf;
    tmplen = len - 1; // 预留结束字符位置。
    tmpval = val;

    /*尝试8进制格式化输出。*/
    do
    {
        tmpbuf[--tmplen] = '0' + (char)(tmpval & 7);
        tmpval >>= 3;

    } while (tmplen);

    /*有余数时表示空间不足，尝试base-256编码输出。*/
    if (tmpval)
    {
        tmpbuf = buf;
        tmplen = len; // 不需要保留结束字符位置。
        tmpval = val;

        memset(tmpbuf, 0, tmplen);

        do
        {
            tmpbuf[--tmplen] = (unsigned char)(tmpval & 0xFF);
            tmpval >>= 8;

        } while (tmplen);

        /*有余数时表示空间不足，返回失败。*/
        if (tmpval)
            return -1;

        /*如果标志位如果被占用，返回失败。*/
        if (*tmpbuf & '\x80')
            return -1;

        /*设置base-256编码标志。*/
        *tmpbuf |= '\x80';
    }

    return 0;
}

int abcdk_tar_char2num(const char *buf, size_t len, uintmax_t *val)
{
    const char *tmpbuf;
    size_t tmplen;
    uintmax_t *tmpval;
    size_t i;

    assert(buf != NULL && len > 0 && val != NULL);

    tmpbuf = buf;
    tmplen = len;
    tmpval = val;

    /*检测是否为base-256编码。*/
    if (*tmpbuf & '\x80')
    {
        /*解码非标志位的数值。*/
        *tmpval = (tmpbuf[i = 0] & '\x3F');

        /*解码其它数据。*/
        for (i += 1; i < len; i++)
        {
            /*检查是否发生数值溢出。*/
            if (*tmpval > (UINTMAX_MAX >> 8))
                return -1;

            *tmpval <<= 8;
            *tmpval |= (unsigned char)(tmpbuf[i]);
        }
    }
    else
    {
        /*跳过不是8进制的数字字符。*/
        for (i = 0; i < len; i++)
        {
            if (abcdk_isodigit(tmpbuf[i]))
                break;
        }

        /*字符转数值。*/
        for (; i < len; i++)
        {
            /*遇到非8进制的数字符时，提前终止。*/
            if (!abcdk_isodigit(tmpbuf[i]))
                break;

            /*检查是否发生数值溢出。*/
            if (*tmpval > (UINTMAX_MAX >> 3))
                return -1;

            *tmpval <<= 3;
            *tmpval |= tmpbuf[i] - '0';
        }

        /*如果提前终止，返回失败。*/
        if (i < len && tmpbuf[i] != '\0')
            return -1;
    }

    return 0;
}

uint32_t abcdk_tar_calc_checksum(abcdk_tar_hdr *hdr)
{
    uint32_t sum = 0;
    int i = 0;

    assert(hdr != NULL);

    for (i = 0; i < 148; ++i)
    {
        sum += ABCDK_PTR2OBJ(uint8_t, hdr, i);
    }

    /*-----跳过checksum(8bytes)字段------*/

    for (i += 8; i < 512; ++i) //...........
    {
        sum += ABCDK_PTR2OBJ(uint8_t, hdr, i);
    }

    sum += 256;

    return sum;
}

uint32_t abcdk_tar_get_checksum(abcdk_tar_hdr *hdr)
{
    uintmax_t val = 0;

    assert(hdr != NULL);

    /*较验和的字段长度8个字节，但只有6个数字，跟着一个NULL(0)，最后一个是空格。*/
    if (abcdk_tar_char2num(hdr->posix.chksum, 7, &val) != 0)
        return -1;

    return val;
}

int64_t abcdk_tar_get_size(abcdk_tar_hdr *hdr)
{
    uintmax_t val = 0;

    assert(hdr != NULL);

    if (abcdk_tar_char2num(hdr->posix.size, sizeof(hdr->posix.size), &val) != 0)
        return -1;

    return val;
}

time_t abcdk_tar_get_mtime(abcdk_tar_hdr *hdr)
{
    uintmax_t val = 0;

    assert(hdr != NULL);

    if (abcdk_tar_char2num(hdr->posix.mtime, sizeof(hdr->posix.mtime), &val) != 0)
        return -1;

    return val;
}

mode_t abcdk_tar_get_mode(abcdk_tar_hdr *hdr)
{
    uintmax_t val = 0;

    assert(hdr != NULL);

    if (abcdk_tar_char2num(hdr->posix.mode, sizeof(hdr->posix.mode), &val) != 0)
        return -1;

    return val;
}

uid_t abcdk_tar_get_uid(abcdk_tar_hdr *hdr)
{
    uintmax_t val = 0;

    assert(hdr != NULL);

    if (abcdk_tar_char2num(hdr->posix.uid, sizeof(hdr->posix.uid), &val) != 0)
        return -1;

    return val;
}

gid_t abcdk_tar_get_gid(abcdk_tar_hdr *hdr)
{
    uintmax_t val = 0;

    assert(hdr != NULL);

    if (abcdk_tar_char2num(hdr->posix.gid, sizeof(hdr->posix.gid), &val) != 0)
        return -1;

    return val;
}

void abcdk_tar_fill(abcdk_tar_hdr *hdr, char typeflag,
                   const char *name, const char *linkname,
                   int64_t size, time_t mtime, mode_t mode)
{
    assert(hdr != NULL && name != NULL);
    assert(size >= 0);
    assert(name[0] != '\0');

    /**/
    hdr->posix.typeflag = typeflag;

    /*Max 99 bytes.*/
    strncpy(hdr->posix.name, name, 99);

    /*Max 99 bytes, may be NULL(0).*/
    if (linkname)
        strncpy(hdr->posix.linkname, linkname, 99);

    strncpy(hdr->posix.magic, TMAGIC, TMAGLEN);
    strncpy(hdr->posix.version, TVERSION, TVERSLEN);

    abcdk_tar_num2char(size, hdr->posix.size, sizeof(hdr->posix.size));
    abcdk_tar_num2char(mtime, hdr->posix.mtime, sizeof(hdr->posix.mtime));
    abcdk_tar_num2char((mode & (S_IRWXU | S_IRWXG | S_IRWXO)), hdr->posix.mode, sizeof(hdr->posix.mode));

    /*较验和的字段长度8个字节，但只有6个数字，跟着一个NULL(0)，最后一个是空格。*/
    memset(hdr->posix.chksum,' ',sizeof(hdr->posix.chksum));
    abcdk_tar_num2char(abcdk_tar_calc_checksum(hdr), hdr->posix.chksum, 7);
}

int abcdk_tar_verify(abcdk_tar_hdr *hdr, const char *magic, size_t size)
{
    uint32_t old_sum = 0;
    uint32_t now_sum = 0;

    assert(hdr != NULL && magic != NULL && size > 0);

    if (abcdk_strncmp(hdr->posix.magic, magic, size, 1) != 0)
        return 0;

    old_sum = abcdk_tar_get_checksum(hdr);
    now_sum = abcdk_tar_calc_checksum(hdr);

    if (old_sum != now_sum)
        return 0;

    return 1;
}

ssize_t abcdk_tar_read(abcdk_tar_t *tar, void *data, size_t size)
{
    assert(tar != NULL && data != NULL && size > 0);

    return abcdk_block_read(tar->fd, data, size, tar->buf);
}

int abcdk_tar_read_align(abcdk_tar_t *tar, size_t size)
{
    char tmp[ABCDK_TAR_BLOCK_SIZE] = {0};
    size_t fix_size;

    assert(tar != NULL && size > 0);

    /*计算需要读取对齐的差额长度。*/
    fix_size = abcdk_align(size, ABCDK_TAR_BLOCK_SIZE) - size;

    /*也许已经对齐。*/
    if (fix_size <= 0)
        return 0;

    return ((abcdk_tar_read(tar, tmp, fix_size) == fix_size) ? 0 : -1);
}

ssize_t abcdk_tar_write(abcdk_tar_t *tar, const void *data, size_t size)
{
    assert(tar != NULL && data != NULL && size > 0);

    return abcdk_block_write(tar->fd, data, size, tar->buf);
}

int abcdk_tar_write_align(abcdk_tar_t *tar, size_t size)
{
    char tmp[ABCDK_TAR_BLOCK_SIZE] = {0};
    size_t fix_size;

    assert(tar != NULL && size > 0);

    /*计算需要写入对齐的差额长度。*/
    fix_size = abcdk_align(size, ABCDK_TAR_BLOCK_SIZE) - size;

    /*也许已经对齐。*/
    if (fix_size <= 0)
        return 0;

    return ((abcdk_tar_write(tar, tmp, fix_size) == fix_size) ? 0 : -1);
}

int abcdk_tar_write_trailer(abcdk_tar_t *tar, uint8_t stuffing)
{
    assert(tar != NULL);

    return abcdk_block_write_trailer(tar->fd, 0, tar->buf);
}

int abcdk_tar_read_hdr(abcdk_tar_t *tar, char name[PATH_MAX], struct stat *attr, char linkname[PATH_MAX])
{
    abcdk_tar_hdr hdr;
    int longnamelen = 0;
    int linknamelen = 0;

    assert(tar != NULL && name != NULL && attr != NULL && linkname != NULL);
    assert(tar->fd >= 0);

    memset(&hdr,0,sizeof(hdr));

    /*完整的头部可能由多个组成，因此可能要多次读取多个头部。*/

again:

    if (abcdk_tar_read(tar, &hdr, ABCDK_TAR_BLOCK_SIZE) != ABCDK_TAR_BLOCK_SIZE)
        goto final_error;

    if (!abcdk_tar_verify(&hdr, TMAGIC, TMAGLEN))
        goto final_error;

    if (hdr.posix.typeflag == ABCDK_USTAR_LONGLINK_TYPE)
    {
        /*长链接名需要特殊处理。*/

        if (abcdk_strncmp(hdr.posix.name, ABCDK_USTAR_LONGNAME_MAGIC, ABCDK_USTAR_LONGNAME_MAGIC_LEN - 1, 1) != 0)
            goto final_error;

        linknamelen = abcdk_tar_get_size(&hdr);
        if (linknamelen <= 0)
            goto final_error;

        if (abcdk_tar_read(tar, linkname, linknamelen) != linknamelen)
            goto final_error;

        if (abcdk_tar_read_align(tar, linknamelen) != 0)
            goto final_error;

        /*头部信息还不完整，继续读取。*/
        goto again;
    }
    else if (hdr.posix.typeflag == ABCDK_USTAR_LONGNAME_TYPE)
    {
        /*长文件名需要特殊处理。*/

        if (abcdk_strncmp(hdr.posix.name, ABCDK_USTAR_LONGNAME_MAGIC, ABCDK_USTAR_LONGNAME_MAGIC_LEN - 1, 1) != 0)
            goto final_error;

        longnamelen = abcdk_tar_get_size(&hdr);
        if (longnamelen <= 0)
            goto final_error;

        if (abcdk_tar_read(tar, name, longnamelen) != longnamelen)
            goto final_error;

        if (abcdk_tar_read_align(tar, longnamelen) != 0)
            goto final_error;

        /*头部信息还不完整，继续读取。*/
        goto again;
    }
    else
    {
        if (REGTYPE == hdr.posix.typeflag || AREGTYPE == hdr.posix.typeflag)
            attr->st_mode = __S_IFREG;
        else if (SYMTYPE == hdr.posix.typeflag)
            attr->st_mode = __S_IFLNK;
        else if (DIRTYPE == hdr.posix.typeflag)
            attr->st_mode = __S_IFDIR;
        else if (CHRTYPE == hdr.posix.typeflag)
            attr->st_mode = __S_IFCHR;
        else if (BLKTYPE == hdr.posix.typeflag)
            attr->st_mode = __S_IFBLK;
        else if (FIFOTYPE == hdr.posix.typeflag)
            attr->st_mode = __S_IFIFO;
        else
            goto final_error;

        attr->st_size = abcdk_tar_get_size(&hdr);
        attr->st_mode |= abcdk_tar_get_mode(&hdr);
        attr->st_ctim.tv_sec = time(NULL);
        attr->st_mtim.tv_sec = abcdk_tar_get_mtime(&hdr);
        attr->st_gid = abcdk_tar_get_gid(&hdr);
        attr->st_uid = abcdk_tar_get_uid(&hdr);

        if (longnamelen <= 0)
        {
            /*name字段的前缀路径，见：POSIX.1-1996 section 10.1.1.*/
            if (hdr.posix.prefix[0])
            {
                strncpy(name, hdr.posix.prefix, sizeof(hdr.posix.prefix));
                abcdk_dirdir(name, "/");
            }

            strncpy(name + strlen(name), hdr.posix.name, sizeof(hdr.posix.name));
        }

        if (linknamelen <= 0)
            strncpy(linkname, hdr.posix.linkname,sizeof(hdr.posix.linkname));
    }

    return 0;

final_error:

    return -1;
}

int abcdk_tar_write_hdr(abcdk_tar_t *tar, const char *name, const struct stat *attr, const char *linkname)
{
    abcdk_tar_hdr hdr;
    int namelen = 0;
    int linknamelen = 0;
    int chk;

    assert(tar != NULL && name != NULL && attr != NULL);
    assert(tar->fd >= 0 && name[0] != '\0');
    assert(S_ISREG(attr->st_mode) || S_ISDIR(attr->st_mode) || S_ISLNK(attr->st_mode));

    memset(&hdr,0,sizeof(hdr));

    /*计算文件名的长度。*/
    namelen = strlen(name);

    /*计算链接名的长度，可能为NULL(0)。*/
    if (linkname)
        linknamelen = strlen(linkname);

    /*链接名的长度大于或等于100时，需要特别处理。*/
    if (linknamelen >= 100)
    {
        /*清空头部准备复用。*/
        memset(&hdr, 0, ABCDK_TAR_BLOCK_SIZE);

        /*填充长链接名的头部信息。*/
        abcdk_tar_fill(&hdr, ABCDK_USTAR_LONGLINK_TYPE, ABCDK_USTAR_LONGNAME_MAGIC, NULL, linknamelen, 0, 0);

        /*长链接名的头部写入到文件。*/
        if (abcdk_tar_write(tar, &hdr, ABCDK_TAR_BLOCK_SIZE) != ABCDK_TAR_BLOCK_SIZE)
            goto final_error;

        /*长链接名写入到文件。*/
        if (abcdk_tar_write(tar, linkname, linknamelen) != linknamelen)
            goto final_error;

        /*也许需要写入对齐。*/
        if (abcdk_tar_write_align(tar, linknamelen) != 0)
            goto final_error;
    }

    /*文件名(包括路径)的长度大于或等于100时，需要特别处理。*/
    if (namelen >= 100)
    {
        /*清空头部准备复用。*/
        memset(&hdr, 0, ABCDK_TAR_BLOCK_SIZE);

        /*填充长文件名的头部信息。*/
        abcdk_tar_fill(&hdr, ABCDK_USTAR_LONGNAME_TYPE, ABCDK_USTAR_LONGNAME_MAGIC, NULL, namelen, 0, 0);

        /*长文件名的头部写入到文件。*/
        if (abcdk_tar_write(tar, &hdr, ABCDK_TAR_BLOCK_SIZE) != ABCDK_TAR_BLOCK_SIZE)
            goto final_error;

        /*长文件名写入到文件。*/
        if (abcdk_tar_write(tar, name, namelen) != namelen)
            goto final_error;

        /*也许需要写入对齐。*/
        if (abcdk_tar_write_align(tar, namelen) != 0)
            goto final_error;
    }

    /*清空头部准备复用。*/
    memset(&hdr, 0, ABCDK_TAR_BLOCK_SIZE);

    /*填充头部信息。*/
    if (S_ISREG(attr->st_mode))
        abcdk_tar_fill(&hdr, REGTYPE, name, NULL, attr->st_size, attr->st_mtim.tv_sec, attr->st_mode);
    else if (S_ISDIR(attr->st_mode))
        abcdk_tar_fill(&hdr, DIRTYPE, name, NULL, 0, attr->st_mtim.tv_sec, attr->st_mode);
    else if (S_ISLNK(attr->st_mode))
        abcdk_tar_fill(&hdr, SYMTYPE, name, linkname, 0, attr->st_mtim.tv_sec, attr->st_mode);

    /*写入到文件。*/
    if (abcdk_tar_write(tar, &hdr, ABCDK_TAR_BLOCK_SIZE) != ABCDK_TAR_BLOCK_SIZE)
        goto final_error;

    return 0;

final_error:

    return -1;
}
