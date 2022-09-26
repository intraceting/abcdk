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
#include "util/uri.h"
#include "util/mmap.h"
#include "http/http.h"

#include "entry.h"

typedef struct _abcdk_test_http
{
    int errcode;
    abcdk_tree_t *args;

    const char *listen;

    abcdk_comm_t *comm;
    abcdk_comm_node_t *listen_node;

} abcdk_test_http_t;

void _abcdk_test_http_accept_cb(abcdk_comm_node_t *node, int *result)
{
    *result = 0;
}

void _abcdk_test_http_event_cb(abcdk_comm_node_t *node,  abcdk_http_request_t *req)
{
    size_t len = 0;
    const char *p,*val;
    for(int i = 0;i<100;i++)
    {
        p = abcdk_http_request_env(req,i);
        if(!p)
            break;

        if(val = abcdk_http_match_env(p,"Content-Length"))
            len = strtol(val,NULL,0);
        
        fprintf(stderr,"%s\n",p);
    }

    if(len>0)
    {
        abcdk_save("./test_http_upload.data",abcdk_http_request_body(req),len,0);
    }

    abcdk_object_t *file = abcdk_mmap2("/home/devel/job/tmp/无标题文档",0,0);

    char buf[1000] = {0};

    sprintf(buf, "HTTP/1.1 %s\r\nConnection: Keep-Alive\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: %lu\r\n\r\n",
            abcdk_http_status_desc(200), file->sizes[0]);

    abcdk_http_response(node,buf,strlen(buf));
    abcdk_http_response2(node,file);
}

void _abcdk_test_http_close_cb(abcdk_comm_node_t *node)
{
    char buf[NAME_MAX]={0};
    
    abcdk_comm_get_sockaddr_str(node,NULL,buf);

    fprintf(stderr,"Disconnect: %s\n",buf);
}

void _abcdk_test_http_work(abcdk_test_http_t *ctx)
{
    abcdk_sockaddr_t addr;
    ctx->listen = abcdk_option_get(ctx->args,"--listen",0,"0.0.0.0:8080");

    ctx->comm = abcdk_comm_start(1,-1);

    ctx->listen_node = abcdk_http_alloc(ctx->comm,100*1024*1024,"/tmp/");
    abcdk_comm_set_userdata(ctx->listen_node,ctx);

    abcdk_sockaddr_from_string(&addr,ctx->listen,1);

    abcdk_http_callback_t cb = {_abcdk_test_http_accept_cb,_abcdk_test_http_event_cb,_abcdk_test_http_close_cb};
    abcdk_http_listen(ctx->listen_node,NULL,&addr,&cb);


    while(getchar() != 'Q')
    {
        sleep(1);
    }

    abcdk_comm_unref(&ctx->listen_node);
    abcdk_comm_stop(&ctx->comm);
}

int abcdk_test_http(abcdk_tree_t *args)
{
    abcdk_test_http_t ctx = {0};

    ctx.args = args;

    _abcdk_test_http_work(&ctx);

    return ctx.errcode;
}