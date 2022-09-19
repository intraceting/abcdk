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
#include <locale.h>
#include "util/general.h"
#include "comm/comm.h"
#include "comm/message.h"
#include "util/uri.h"
#include "entry.h"

typedef struct _abcdk_test_httpd
{
    int errcode;
    abcdk_tree_t *args;

    const char *listen;

    abcdk_comm_t *comm;
    abcdk_comm_node_t *listen_node;

} abcdk_test_httpd_t;


void _abcdk_test_httpd_accept(abcdk_comm_node_t *node)
{
   // fprintf(stderr,"aaaa\n");
}

void _abcdk_test_httpd_connect(abcdk_comm_node_t *node)
{
    /*已连接到远端，注册读写事件。*/
    abcdk_comm_recv_watch(node);
}


void _abcdk_test_httpd_input(abcdk_comm_node_t *node)
{
    while (1)
    {
        char buf[101] = {0};
        ssize_t r = abcdk_comm_recv(node, buf, 100);
        if (r <= 0)
        {
            abcdk_comm_recv_watch(node);
            return;
        }

        fprintf(stderr, "%s", buf);
        fflush(stderr);
    }
}

void _abcdk_test_httpd_close(abcdk_comm_node_t *node)
{
    char sockname[NAME_MAX] = {0}, peername[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(node,sockname,peername);

    fprintf(stderr,"Disconnect: %s->%s\n",sockname,peername);
}

void _abcdk_test_httpd_event_cb(abcdk_comm_node_t *node, uint32_t event, abcdk_comm_node_t *listen)
{
    if(event == ABCDK_COMM_EVENT_INPUT)
        _abcdk_test_httpd_input(node);
    else if(event == ABCDK_COMM_EVENT_CONNECT)
        _abcdk_test_httpd_connect(node);
    else if(event == ABCDK_COMM_EVENT_ACCEPT)
        _abcdk_test_httpd_accept(node);
    else if(event == ABCDK_COMM_EVENT_CLOSE)
        _abcdk_test_httpd_close(node);
}

void _abcdk_test_httpd_work(abcdk_test_httpd_t *ctx)
{
    abcdk_sockaddr_t addr;
    ctx->listen = abcdk_option_get(ctx->args,"--listen",0,"0.0.0.0:8080");

    ctx->comm = abcdk_comm_start(1);

    ctx->listen_node = abcdk_comm_node_alloc(ctx->comm);
    abcdk_comm_set_userdata(ctx->listen_node,ctx);

    abcdk_sockaddr_from_string(&addr,ctx->listen,1);
    abcdk_comm_listen(ctx->listen_node,NULL,&addr,_abcdk_test_httpd_event_cb);


    while(getchar() != 'Q')
    {
        sleep(1);
    }

    abcdk_comm_node_unref(&ctx->listen_node);
    abcdk_comm_stop(&ctx->comm);
}

int abcdk_test_httpd(abcdk_tree_t *args)
{
    abcdk_test_httpd_t ctx = {0};

    ctx.args = args;

    _abcdk_test_httpd_work(&ctx);

    return ctx.errcode;
}