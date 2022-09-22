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
#include "log/log.h"
#include "comm/easy.h"
#include "entry.h"


void test_easy_request_cb(abcdk_comm_node_t *easy, const void *data, size_t len)
{
    char sockname_str[NAME_MAX] = {0}, peername_str[NAME_MAX] = {0};

    abcdk_comm_get_sockaddr_str(easy,sockname_str,peername_str);

    printf("Server(%s -> %s): ", sockname_str, peername_str);

    if(!data)
    {
        printf(" Disconnected.\n");
    }
    else
    {
            uint64_t a = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6);
            uint64_t b = atoll((char*)data);

           printf("%lu-%lu=%lu",a,b,a-b);

        usleep(rand()%10000+1000);

        abcdk_comm_easy_response(easy,data,len);
        abcdk_comm_easy_request(easy,data,len,NULL);


    }

     printf("\n");
}

void test_easy_request2_cb(abcdk_comm_node_t *easy, const void *data, size_t len)
{
    char sockname_str[NAME_MAX] = {0}, peername_str[NAME_MAX] = {0};
    
    abcdk_comm_get_sockaddr_str(easy,sockname_str,peername_str);

    printf("Client(%s -> %s): ", sockname_str, peername_str);

    if(!data)
    {
        printf(" Disconnected.");
    }
    else
    {
     //   printf(" %s\n",(char*)data);
    }

     printf("\n");
}


int abcdk_test_easy(abcdk_tree_t *args)
{
    signal(SIGPIPE,NULL);

    abcdk_comm_t *ctx = abcdk_comm_start(0,-1);

    SSL_CTX *server_ssl_ctx = NULL;
    SSL_CTX *client_ssl_ctx[4] = {NULL};

#ifdef HAVE_OPENSSL

    const char *capath = abcdk_option_get(args,"--ca-path",0,NULL);

    if (capath)
    {
        server_ssl_ctx = abcdk_openssl_ssl_ctx_alloc(1, NULL, capath, 2);

        abcdk_openssl_ssl_ctx_load_crt(server_ssl_ctx, abcdk_option_get(args, "--crt-file", 0, NULL),
                                       abcdk_option_get(args, "--key-file", 0, NULL),
                                       abcdk_option_get(args, "--key-pwd", 0, NULL));

   //     SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);

        SSL_CTX_set_verify(server_ssl_ctx, SSL_VERIFY_PEER, NULL);

        for(int i =0;i<4;i++)
        {
            client_ssl_ctx[i] = abcdk_openssl_ssl_ctx_alloc(0, NULL, capath, 2);

        abcdk_openssl_ssl_ctx_load_crt(client_ssl_ctx[i], abcdk_option_get(args, "--crt2-file", i, NULL),
                                       abcdk_option_get(args, "--key2-file", i, NULL),
                                       abcdk_option_get(args, "--key2-pwd", i, NULL));

            SSL_CTX_set_verify(client_ssl_ctx[i], SSL_VERIFY_PEER, NULL);
        }

    }
#endif //HAVE_OPENSSL

    const char *sunpath = "/tmp/test_easy.sock";
    unlink(sunpath);

    abcdk_sockaddr_t addr = {0};
    abcdk_sockaddr_t addr2 = {0};

    const char *listen_p = abcdk_option_get(args,"--listen",0,"0.0.0.0:12345");
    abcdk_sockaddr_from_string(&addr,listen_p,0);
    //addr.family = AF_UNIX;
    //strncpy(addr.addr_un.sun_path,sunpath,108);

    abcdk_comm_node_t *easy_listen = abcdk_comm_easy_alloc(ctx,3333);
    abcdk_comm_easy_listen(easy_listen,server_ssl_ctx,&addr,test_easy_request_cb);

    const char *connect_p = abcdk_option_get(args,"--connect",0,"127.0.0.1:12345");
    abcdk_sockaddr_from_string(&addr2,connect_p,0);
    //addr2.family = AF_UNIX;
    //strncpy(addr2.addr_un.sun_path,sunpath,108);

    int nn = 4;
    abcdk_comm_node_t *easy_client[40] = {NULL};
    for (int i = 0; i < nn; i++)
    {
        easy_client[i] = abcdk_comm_easy_alloc(ctx,3333);
        abcdk_comm_easy_connect(easy_client[i],client_ssl_ctx[i], &addr2, test_easy_request2_cb);
    }

    uint64_t d = 0,s = 0;
    s = abcdk_clock(d,&d);

    

    #pragma omp parallel for num_threads(nn)
    for(int i = 0;i<100000;i++)
    {
#ifdef _OPENMP
        omp_get_thread_num();
#endif
        
        uint64_t d = 0,s = 0;
        s = abcdk_clock(d,&d);

        int len = 1000000;
        char *req= (char*)abcdk_heap_alloc(len);
        abcdk_comm_message_t *rsp= NULL;

        sprintf(req,"%lu",abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6));

        abcdk_comm_easy_request(easy_client[i%nn],req,len,&rsp);
        

        if (rsp)
        {

            // printf("%d=%s\n",i,(char*)abcdk_comm_message_data(rsp));

            abcdk_comm_message_unref(&rsp);
        }
        else
        {
            printf("Pipe(%d) %s timeout\n",i%4,req);
        }

        abcdk_heap_free(req);

        s = abcdk_clock(d,&d);

      //  printf("[%d]:s = %lu,d = %lu\n",i,s,d);
    }

    s = abcdk_clock(d,&d);

    printf("s = %lu,d = %lu\n",s,d);

 //   abcdk_comm_easy_set_timeout(easy_listen,1);

  //  abcdk_comm_easy_unref(&easy_listen);
    while (getchar() != 'Q')
        ;

    for(int i = 0;i<nn;i++)
        abcdk_comm_unref(&easy_client[i]);


    abcdk_comm_stop(&ctx);

    return 0;
}