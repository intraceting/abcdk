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
#include "abcdkutil/general.h"
#include "abcdkutil/getargs.h"
#include "abcdkutil/scsi.h"
#include "abcdkutil/mtx.h"

/**/
enum _abcdkmtx_cmd
{
    /** 枚举磁带库所有元素。*/
    ABCDKMTX_STATUS = 1,
#define ABCDKMTX_STATUS ABCDKMTX_STATUS

    /** 移动磁带。*/
    ABCDKMTX_MOVE = 2
#define ABCDKMTX_MOVE ABCDKMTX_MOVE

};

void _abcdkmtx_print_usage(abcdk_tree_t *args, int only_version)
{
    char name[NAME_MAX] = {0};

    /*Clear errno.*/
    errno = 0;

    abcdk_proc_basename(name);

#ifdef BUILD_VERSION_DATETIME
    fprintf(stderr, "\n%s Build %s\n", name, BUILD_VERSION_DATETIME);
#endif //BUILD_VERSION_DATETIME

    fprintf(stderr, "\n%s Version %d.%d\n", name, ABCDK_VERSION_MAJOR, ABCDK_VERSION_MINOR);

    if (only_version)
        return;

    fprintf(stderr, "\nOptions:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\tShow this help message and exit.\n");

    fprintf(stderr, "\n\t--version\n");
    fprintf(stderr, "\t\tOutput version information and exit.\n");

    fprintf(stderr, "\n\t--dev < FILE >\n");
    fprintf(stderr, "\t\tGeneric SCSI device.\n");

    fprintf(stderr, "\n\t--src < ADDRESS >\n");
    fprintf(stderr, "\t\tSource Element Address.\n");

    fprintf(stderr, "\n\t--dst < ADDRESS >\n");
    fprintf(stderr, "\t\tDestination Element Address.\n");

    fprintf(stderr, "\n\t--exclude-barcode\n");
    fprintf(stderr, "\t\tExclude BARCODE. default: include\n");

    fprintf(stderr, "\n\t--exclude-dvcid\n");
    fprintf(stderr, "\t\tExclude DVCID. default: include\n");

    fprintf(stderr, "\n\t--cmd < NUMBER >\n");
    fprintf(stderr, "\t\tCommand. default: %d\n", ABCDKMTX_STATUS);

    fprintf(stderr, "\n\t\t%d: Report elements.\n", ABCDKMTX_STATUS);
    fprintf(stderr, "\t\t%d: Move Medium.\n", ABCDKMTX_MOVE);
}

static struct _abcdkmtx_sense_dict
{   
    uint8_t key;
    uint8_t asc;
    uint8_t ascq;
    const char *msg;
}abcdkmtx_sense_dict[] = {
    /*KEY=0x00*/
    {0x00, 0x00, 0x00, "No Sense"},
    /*KEY=0x01*/
    {0x01, 0x00, 0x00, "Recovered Error"},
    /*KEY=0x02*/
    {0x02, 0x00, 0x00, "Not Ready"},
    /*KEY=0x03*/
    {0x03, 0x00, 0x00, "Medium Error"},
    /*KEY=0x04*/
    {0x04, 0x00, 0x00, "Hardware Error"},
    /*KEY=0x05*/
    {0x05, 0x00, 0x00, "Illegal Request"},
    {0x05, 0x21, 0x01, "Invalid element address"},
    {0x05, 0x24, 0x00, "Invalid field CDB or address"},
    {0x05, 0x3b, 0x0d, "Medium destination element full"},
    {0x05, 0x3b, 0x0e, "Medium source element empty"},
    {0x05, 0x53, 0x02, "Library media removal prevented state set"},
    {0x05, 0x53, 0x03, "Drive media removal prevented state set"},
    {0x05, 0x44, 0x80, "Bad status library controller"},
    {0x05, 0x44, 0x81, "Source not ready"},
    {0x05, 0x44, 0x82, "Destination not ready"},
    /*KEY=0x06*/
    {0x06, 0x00, 0x00, "Unit Attention"},
    /*KEY=0x0b*/
    {0x0b, 0x00, 0x00, "Command Aborted"}
};

void _abcdkmtx_printf_sense(abcdk_scsi_io_stat *stat)
{
    uint8_t key = 0, asc = 0, ascq = 0;
    const char *msg_p = "Unknown";

    key = abcdk_scsi_sense_key(stat->sense);
    asc = abcdk_scsi_sense_code(stat->sense);
    ascq = abcdk_scsi_sense_qualifier(stat->sense);

    for (size_t i = 0; i < ABCDK_ARRAY_SIZE(abcdkmtx_sense_dict); i++)
    {
        if (abcdkmtx_sense_dict[i].key != key)
            continue;

        msg_p = abcdkmtx_sense_dict[i].msg;

        if (abcdkmtx_sense_dict[i].asc != asc || abcdkmtx_sense_dict[i].ascq != ascq)
            continue;

        msg_p = abcdkmtx_sense_dict[i].msg;
        break;
    }

    syslog(LOG_INFO, "Sense(KEY=%02X,ASC=%02X,ASCQ=%02X): %s.", key, asc, ascq, msg_p);
}

int _abcdkmtx_printf_elements_cb(size_t deep, abcdk_tree_t *node, void *opaque)
{
    if (deep == 0)
    {
        abcdk_tree_fprintf(stdout, deep, node, "%s\n", node->alloc->pptrs[0]);
    }
    else
    {
        abcdk_tree_fprintf(stdout, deep, node, "%-6hu\t|%-2hhu\t|%-2hhu\t|%-10s\t|%-10s\t|\n",
                          ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0),
                          ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0),
                          ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ISFULL], 0),
                          node->alloc->pptrs[ABCDK_MTX_ELEMENT_BARCODE],
                          node->alloc->pptrs[ABCDK_MTX_ELEMENT_DVCID]);
    }

    return 1;
}

void _abcdkmtx_printf_elements(abcdk_tree_t *root)
{
    abcdk_tree_iterator_t it = {0, _abcdkmtx_printf_elements_cb, NULL};
    abcdk_tree_scan(root, &it);
}

int _abcdkmtx_find_changer_cb(size_t deep, abcdk_tree_t *node, void *opaque)
{
    uint16_t *t_p = (uint16_t *)opaque;

    if (deep == 0)
        return 1;

    if (ABCDK_PTR2U8(node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0) != ABCDK_MXT_ELEMENT_CHANGER)
        return 1;

    *t_p = ABCDK_PTR2U16(node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0);

    return -1;
}

uint16_t _abcdkmtx_find_changer(abcdk_tree_t *root)
{
    uint16_t t = 0;

    abcdk_tree_iterator_t it = {0, _abcdkmtx_find_changer_cb, &t};
    abcdk_tree_scan(root, &it);

    return t;
}

void _abcdkmtx_move_medium(abcdk_tree_t *args, int fd, abcdk_tree_t *root)
{
    abcdk_scsi_io_stat stat = {0};
    int t = 0, s = 65536, d = 65536;
    int chk;

    t = _abcdkmtx_find_changer(root);
    s = abcdk_option_get_int(args, "--src", 0, 65536);
    d = abcdk_option_get_int(args, "--dst", 0, 65536);

    chk = abcdk_mtx_move_medium(fd, t, s, d, 1800 * 1000, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EINVAL,print_sense);

    /*No error.*/
    goto final;

print_sense:

    _abcdkmtx_printf_sense(&stat);

final:

    return;
}

void _abcdkmtx_work(abcdk_tree_t *args)
{
    abcdk_scsi_io_stat stat = {0};
    abcdk_tree_t *root = NULL;
    uint8_t type = 0;
    char vendor[32] = {0};
    char product[64] = {0};
    char sn[64] = {0};
    int fd = -1;
    const char *dev_p = NULL;
    int voltag = 1;
    int dvcid = 1;
    int cmd = 0;
    int chk;

    /*Clear errno.*/
    errno = 0;

    dev_p = abcdk_option_get(args, "--dev", 0, "");
    voltag = (abcdk_option_exist(args, "--exclude-barcode") ? 0 : 1);
    dvcid = (abcdk_option_exist(args, "--exclude-dvcid") ? 0 : 1);
    cmd = abcdk_option_get_int(args, "--cmd", 0, ABCDKMTX_STATUS);
    

    if (access(dev_p, F_OK) != 0)
    {
        syslog(LOG_WARNING, "'%s' No such device.", dev_p);
        goto final;
    }

    root = abcdk_tree_alloc3(200);
    if (!root)
        goto final;

    fd = abcdk_open(dev_p, 0, 0, 0);
    if (fd < 0)
    {
        syslog(LOG_WARNING, "'%s' %s.",dev_p,strerror(errno));
        goto final;
    }

    chk = abcdk_scsi_inquiry_standard(fd, &type, vendor, product, 3000, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    if (type != TYPE_MEDIUM_CHANGER)
    {
        syslog(LOG_WARNING, "'%s' Not Medium Changer.", dev_p);
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    chk = abcdk_scsi_inquiry_serial(fd, NULL, sn, 3000, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    snprintf(root->alloc->pptrs[0], root->alloc->sizes[0], "%s(%s-%s)", sn, vendor, product);

    chk = abcdk_mtx_inquiry_element_status(root, fd, voltag,dvcid,-1, &stat);
    if (chk != 0 || stat.status != GOOD)
        ABCDK_ERRNO_AND_GOTO1(EPERM,print_sense);

    if (cmd == ABCDKMTX_STATUS)
    {
        _abcdkmtx_printf_elements(root);
    }
    else if (cmd == ABCDKMTX_MOVE)
    {
        _abcdkmtx_move_medium(args, fd, root);
    }
    else
    {
        syslog(LOG_WARNING, "Not supported.");
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);
    }

    /*No error.*/
    goto final;

print_sense:

    _abcdkmtx_printf_sense(&stat);

final:

    abcdk_closep(&fd);
    abcdk_tree_free(&root);
}

int main(int argc, char **argv)
{
    abcdk_tree_t *args;

    args = abcdk_tree_alloc3(1);
    if (!args)
        goto final;

    abcdk_getargs(args, argc, argv, "--");

    abcdk_openlog(NULL, LOG_INFO, 1);

    if (abcdk_option_exist(args, "--help"))
    {
        _abcdkmtx_print_usage(args, 0);
    }
    else if (abcdk_option_exist(args, "--version"))
    {
        _abcdkmtx_print_usage(args, 1);
    }
    else
    {
        _abcdkmtx_work(args);
    }

final:

    abcdk_tree_free(&args);

    return errno;
}