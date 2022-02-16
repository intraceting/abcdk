/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "shell/scsi.h"

int _abcdk_scsi_get_type(const char* path,uint32_t *type)
{
    char buf[20] = {0};
    ssize_t rlen = 0;
    char file[PATH_MAX] = {0};
    int chk;

    abcdk_dirdir(file,path);
    abcdk_dirdir(file,"type");

    rlen = abcdk_load(file, buf, 19, 0);
    if (rlen <= 0)
        return -1;

    abcdk_strtrim(buf,isspace,2);
    chk = abcdk_strtype(buf, isdigit);
    if (chk == 0)
        return -1;

    /*字符串转数值。*/
    *type = atoi(buf);

    return 0;
}

int _abcdk_scsi_get_serial(const char* path,char serial[NAME_MAX])
{
    char buf[255] = {0};
    ssize_t rlen = 0;
    char file[PATH_MAX] = {0};

    abcdk_dirdir(file,path);
    abcdk_dirdir(file,"vpd_pg80");

    rlen = abcdk_load(file, buf, 255, 0);
    if (rlen <= 0)
    return -1;

    memcpy(serial, buf + 4, buf[3]);
    abcdk_strtrim(serial, isspace, 2);

    return 0;
}

int _abcdk_scsi_get_vendor(const char* path,char vendor[NAME_MAX])
{
    ssize_t rlen = 0;
    char file[PATH_MAX] = {0};

    abcdk_dirdir(file,path);
    abcdk_dirdir(file,"vendor");

    rlen = abcdk_load(file, vendor, NAME_MAX-1,0);
    if (rlen <= 0)
        return -1;

    abcdk_strtrim(vendor,isspace,2);

    return 0;
}

int _abcdk_scsi_get_model(const char* path,char model[NAME_MAX])
{
    ssize_t rlen = 0;
    char file[PATH_MAX] = {0};

    abcdk_dirdir(file,path);
    abcdk_dirdir(file,"model");

    rlen = abcdk_load(file, model, NAME_MAX-1, 0);
    if (rlen <= 0)
        return -1;

    abcdk_strtrim(model,isspace,2);
    
    return 0;
}

int _abcdk_scsi_get_revision(const char* path,char revision[NAME_MAX])
{
    ssize_t rlen = 0;
    char file[PATH_MAX] = {0};

    abcdk_dirdir(file,path);
    abcdk_dirdir(file,"rev");

    rlen = abcdk_load(file, revision, NAME_MAX-1, 0);
    if (rlen <= 0)
        return -1;

    abcdk_strtrim(revision,isspace,2);
    
    return 0;
}

int _abcdk_scsi_get_devname(const char *path,int type, char devname[NAME_MAX])
{
    char buf[PATH_MAX] = {0};
    char buf2[PATH_MAX] = {0};
    char buf3[40] = {0};
    int deflag = 0,major = -1,minor = -1;
    abcdk_tree_t *dir = NULL;
    char path2[PATH_MAX] = {0};
    int chk;

    abcdk_dirdir(path2, path);
    if(type == TYPE_DISK)
        abcdk_dirdir(path2, "block");
    else if(type == TYPE_TAPE)
        abcdk_dirdir(path2, "scsi_tape");
    if(type == TYPE_ROM)
        abcdk_dirdir(path2, "block");
    else if(type == TYPE_MEDIUM_CHANGER)
        abcdk_dirdir(path2, "scsi_changer");

    dir = abcdk_tree_alloc3(1);
    if (!dir)
        return -1;

    chk = abcdk_dirent_open(dir, path2);
    if (chk != 0)
        goto final;

    while (1)
    {
        memset(buf, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir, buf);
        if (chk != 0)
            break;

        /*暂存设备名字。*/
        memset(devname, 0, NAME_MAX);
        abcdk_dirdir(devname, "/dev/");
        abcdk_basename(devname + strlen(devname), buf);

        /*检查/dev/下是否存在。*/
        if (access(devname, F_OK) != 0)
            continue;

        /*获取设备号(主，次)。*/
        major = -1, minor = -1;
        memset(buf2, 0, PATH_MAX);

        abcdk_dirdir(buf2, buf);
        abcdk_dirdir(buf2, "dev");

        abcdk_load(buf2, buf3, 39, 0);
        sscanf(buf3, "%d:%d", &major, &minor);

        if (type == TYPE_TAPE)
        {
            /* https://www.kernel.org/doc/Documentation/scsi/st.txt */
            
            memset(buf2, 0, PATH_MAX);
            abcdk_dirdir(buf2, buf);
            abcdk_dirdir(buf2, "defined");
            abcdk_load(buf2,buf3,39,0);

            /*只查找已经定义的设备。*/
            deflag = 0;
            sscanf(buf3, "%d", &deflag);
            if (deflag == 0)
                continue;

            /*次设备号 >= 128，关闭设备后不执行自动倒带。*/
            if (minor >= 128)
                break;
        }
        else if (type == TYPE_MEDIUM_CHANGER)
        {
            /* https://www.kernel.org/doc/Documentation/scsi/scsi-changer.txt */
            if (major == 86 && minor == 0)
                break;
        }
        else
        {
            break;
        }

    }

    /*No error.*/
    chk = 0;

final:

    abcdk_tree_free(&dir);

    return chk;
}

int _abcdk_scsi_get_generic(const char *path, char generic[NAME_MAX])
{
    char buf[PATH_MAX] = {0};
    char buf2[PATH_MAX] = {0};
    abcdk_tree_t *dir = NULL;
    char path2[PATH_MAX] = {0};
    int chk;

    abcdk_dirdir(path2, path);
    abcdk_dirdir(path2, "scsi_generic");

    dir = abcdk_tree_alloc3(1);
    if (!dir)
        return -1;

    chk = abcdk_dirent_open(dir, path2);
    if (chk != 0)
        goto final;

    while (1)
    {
        memset(buf, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir, buf);
        if (chk != 0)
            break;

        /*暂存设备名字。*/
        memset(generic, 0, NAME_MAX);
        abcdk_dirdir(generic, "/dev/");
        abcdk_basename(generic + strlen(generic), buf);

        /*检查/dev/下是否存在。*/
        if (access(generic, F_OK) != 0)
            continue;
    }

    /*No error.*/
    chk = 0;

final:

    abcdk_tree_free(&dir);

    return chk;
}

void abcdk_scsi_list(abcdk_tree_t *list)
{
    abcdk_tree_t *dev = NULL;
    abcdk_scsi_info_t *dev_p = NULL;
    abcdk_tree_t *dir = NULL;
    char path[PATH_MAX] = {0};
    uint32_t type;
    int chk;

    assert(list != NULL);

    dir = abcdk_tree_alloc3(1);
    if (!dir)
        goto final;

    chk = abcdk_dirent_open(dir, "/sys/bus/scsi/devices");
    if (chk != 0)
        goto final;

    while (1)
    {
        memset(path, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir, path);
        if (chk != 0)
            break;

        /*跳过无法获取类型的设备。*/
        chk = _abcdk_scsi_get_type(path,&type);
        if (chk != 0)
            continue;

        dev = abcdk_tree_alloc3(sizeof(abcdk_scsi_info_t));
        if(!dev)
            break;

        dev_p = (abcdk_scsi_info_t*)dev->alloc->pptrs[0];
        abcdk_tree_insert2(list,dev,0);

        abcdk_basename(dev_p->bus,path);
        dev_p->type = type;
        _abcdk_scsi_get_serial(path,dev_p->serial);
        _abcdk_scsi_get_vendor(path,dev_p->vendor);
        _abcdk_scsi_get_model(path,dev_p->model);
        _abcdk_scsi_get_revision(path,dev_p->revision);
        _abcdk_scsi_get_devname(path,type,dev_p->devname);
        _abcdk_scsi_get_generic(path,dev_p->generic);
    }

final:

    abcdk_tree_free(&dir);

    return;
}