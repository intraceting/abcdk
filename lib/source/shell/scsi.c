/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/shell/scsi.h"

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

int _abcdk_scsi_get_serial_from_dev(const char* dev,char serial[NAME_MAX])
{
    abcdk_scsi_io_stat_t stat;
    int fd;
    int chk;

    fd = abcdk_open(dev, 0, 0, 0);
    if (fd < 0)
        return -1;

    chk = abcdk_scsi_inquiry_serial(fd, NULL, serial, 3000, &stat);
    if (chk != 0 || stat.status != GOOD)
        chk = -2;

    abcdk_closep(&fd);

    return chk;
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
    else if(type == TYPE_ROM)
        abcdk_dirdir(path2, "block");
    else if(type == TYPE_MEDIUM_CHANGER)
        abcdk_dirdir(path2, "scsi_changer");
    else 
        return -1;

    dir = abcdk_tree_alloc3(1);
    if (!dir)
        return -1;

    chk = abcdk_dirent_open(&dir, path2);
    if (chk != 0)
        goto final;

    while (1)
    {
        memset(buf, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir,NULL, buf,1);
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

    chk = abcdk_dirent_open(&dir, path2);
    if (chk != 0)
        goto final;

    while (1)
    {
        memset(buf, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir,NULL, buf,1);
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

int abcdk_scsi_get_info(const char *path, abcdk_scsi_info_t *info)
{
    int chk;

    assert(path != NULL && info != NULL);

    chk = _abcdk_scsi_get_type(path, &info->type);
    if (chk != 0)
        return -1;

    chk = _abcdk_scsi_get_revision(path, info->revision);
    if (chk != 0)
        return -1;

    chk = _abcdk_scsi_get_vendor(path, info->vendor);
    if (chk != 0)
        return -1;

    chk = _abcdk_scsi_get_model(path, info->model);
    if (chk != 0)
        return -1;

    _abcdk_scsi_get_serial(path, info->serial);
    _abcdk_scsi_get_devname(path, info->type, info->devname);
    _abcdk_scsi_get_generic(path, info->generic);

    /*尝试通过设备获取，但需要具备相应的权限。*/
    if(info->serial[0] == '\0')
        _abcdk_scsi_get_serial_from_dev(info->generic,info->serial);

    return 0;
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
        return;

    chk = abcdk_dirent_open(&dir, "/sys/bus/scsi/devices/");
    if (chk != 0)
        goto final;

    /*遍历目录。*/
    while (1)
    {
        memset(path, 0, PATH_MAX);
        chk = abcdk_dirent_read(dir,NULL, path,1);
        if (chk != 0)
            break;

        /*如果无法识别类型则跳过。*/
        chk = _abcdk_scsi_get_type(path,&type);
        if(chk != 0)
            continue;

        dev = abcdk_tree_alloc3(sizeof(abcdk_scsi_info_t));
        if(!dev)
            break;

        dev_p = (abcdk_scsi_info_t*)dev->obj->pptrs[0];
        abcdk_tree_insert2(list,dev,0);

        /*从路径中分离bus并保存。*/
        abcdk_basename(dev_p->bus,path);
        /*提取其它字段并保存。*/
        abcdk_scsi_get_info(path,dev_p);
    }

final:

    abcdk_tree_free(&dir);

    return;
}

int _abcdk_scsi_find(abcdk_tree_t *list, abcdk_tree_t *node)
{
    abcdk_tree_t *p;
    abcdk_scsi_info_t *dev_p, *dev_q;

    /*链表为空，直接返回“未找到”。*/
    if (!list)
        return 0;

    dev_q = (abcdk_scsi_info_t *)node->obj->pptrs[0];

    p = abcdk_tree_child(list, 1);
    while (p)
    {
        dev_p = (abcdk_scsi_info_t *)p->obj->pptrs[0];

        if (abcdk_strcmp(dev_p->bus, dev_q->bus,1) == 0 &&
            abcdk_strcmp(dev_p->devname, dev_q->devname,1) == 0 &&
            abcdk_strcmp(dev_p->generic, dev_q->generic,1) == 0)
        {
            return 1;
        }

        p = abcdk_tree_sibling(p, 0);
    }

    return 0;
}

void _abcdk_scsi_diff(abcdk_tree_t *old_list,abcdk_tree_t *new_list,abcdk_tree_t **diff,int add)
{
    abcdk_tree_t *p;
    abcdk_tree_t *tmp;
    abcdk_tree_t *diff_p;

    abcdk_tree_free(diff);
    *diff = abcdk_tree_alloc3(1);
    if (!*diff)
        return;

    diff_p = *diff;

    /*
     * add == 1 : 从旧的中查找新的。
     * del == 0 ; 从新的中查找旧的。
    */
   
    p = abcdk_tree_child((add ? new_list : old_list), 1);
    while (p)
    {
        if (!_abcdk_scsi_find((add ? old_list : new_list), p))
        {
            tmp = abcdk_tree_alloc(abcdk_object_refer(p->obj));//增加引用计数。
            if (!tmp)
                return;

            abcdk_tree_insert2(diff_p, tmp, 0);
        }

        p = abcdk_tree_sibling(p, 0);
    }

    /*如果未发生变化，删除差异链表。*/
    if(!abcdk_tree_child(diff_p,1))
        abcdk_tree_free(diff);

    return;
}

void _abcdk_scsi_check_ok(abcdk_tree_t *list)
{
    abcdk_tree_t *p,*tmp;
    abcdk_scsi_info_t *dev_p;

    p = abcdk_tree_child(list, 1);
    while (p)
    {
        dev_p = (abcdk_scsi_info_t *)p->obj->pptrs[0];

        if (dev_p->devname[0] == '\0' || dev_p->generic[0] == '\0')
        {
            /*下一个节点。*/
            p = abcdk_tree_sibling(tmp = p, 0);

            /*删除不完成的节点。*/
            abcdk_tree_unlink(tmp);
            abcdk_tree_free(&tmp);
        }
        else 
        {
            /*下一个节点。*/
            p = abcdk_tree_sibling(p, 0);
        }
    }
}

void abcdk_scsi_watch(abcdk_tree_t **snapshot, abcdk_tree_t **add, abcdk_tree_t **del)
{
    abcdk_tree_t *tmp = NULL;

    assert(snapshot != NULL);

    tmp = abcdk_tree_alloc3(1);
    if (!tmp)
        return;

    abcdk_scsi_list(tmp);
    _abcdk_scsi_check_ok(tmp);

    if(*snapshot)
    {
        if(add)
            _abcdk_scsi_diff(*snapshot,tmp,add,1);
        if(del)
            _abcdk_scsi_diff(*snapshot,tmp,del,0);
    }
    else
    {
        if(add)
            _abcdk_scsi_diff(NULL,tmp,add,1);
    }
    
    abcdk_tree_free(snapshot);
    _abcdk_scsi_diff(NULL,tmp,snapshot,1);
    
    abcdk_tree_free(&tmp);
}
