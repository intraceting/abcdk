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
#include <sys/sendfile.h>
#include "abcdk/general.h"
#include "abcdk/getargs.h"
#include "abcdk/clock.h"
#include "abcdk/thread.h"
#include "abcdk/signal.h"
#include "abcdkcomm/mux.h"

void* sigwaitinfo_cb(void* args)
{
    abcdk_signal_t sig;
    sigfillset(&sig.signals);

    sigdelset(&sig.signals, SIGKILL);
    sigdelset(&sig.signals, SIGSEGV);
    sigdelset(&sig.signals, SIGSTOP);

    sig.signal_cb = NULL;
    sig.opaque = NULL;
     
    abcdk_sigwaitinfo(&sig,-1);

    return NULL;
}

void* server_loop(void* args)
{
    abcdk_mux_t *m = (abcdk_mux_t *)args;
    static volatile pthread_t leader = 0;
    static int l = -1;

    if (abcdk_thread_leader_test(&leader) == 0)
    {

        l = abcdk_socket(ABCDK_IPV4, 0);
        abcdk_sockaddr_t a = {0};
        abcdk_sockaddr_from_string(&a, "localhost:12345", 1);

        int flag = 1;
        abcdk_sockopt_option_int(l, SOL_SOCKET, SO_REUSEPORT, &flag, 2);
        abcdk_sockopt_option_int(l, SOL_SOCKET, SO_REUSEADDR, &flag, 2);

        abcdk_bind(l, &a);
        listen(l, SOMAXCONN);

        assert(abcdk_mux_attach2(m, l, 0) == 0);
        assert(abcdk_mux_mark(m, l, ABCDK_EPOLL_INPUT, 0) == 0);
    }

    while(1)
    {
        abcdk_epoll_event e;
        int chk = abcdk_mux_wait(m, &e, -1);
        if (chk < 0)
            break;

        if(e.events & ABCDK_EPOLL_ERROR)
        {
            assert(abcdk_mux_mark(m,e.data.fd,0,e.events)==0);
            assert(abcdk_mux_unref(m,e.data.fd,e.events)==0);
            assert(abcdk_mux_detach(m,e.data.fd)==0);
            abcdk_closep(&e.data.fd);
        }
        else if(e.data.fd == l)
        {
            while(1)
            {
                int c = abcdk_accept(e.data.fd,NULL);
                if(c<0)
                    break;

                int flag=1;
                assert(abcdk_sockopt_option_int(c, IPPROTO_TCP, TCP_NODELAY,&flag, 2) == 0);

                assert(abcdk_mux_attach2(m, c,10*1000) == 0);
                assert(abcdk_mux_mark(m,c,ABCDK_EPOLL_INPUT,0)==0);
            }
            
            assert(abcdk_mux_mark(m,e.data.fd,ABCDK_EPOLL_INPUT,ABCDK_EPOLL_INPUT)==0);
        }
        else
        {
            if(e.events & ABCDK_EPOLL_INPUT)
            {
                while (1)
                {
                    char buf[100];
                    int r = recv(e.data.fd, buf, 100, 0);
                    if (r > 0)
                    {
                        printf("%s\n",buf);
                        continue;
                    }
                    if (r == -1 && errno == EAGAIN)
                        assert(abcdk_mux_mark(m,e.data.fd, ABCDK_EPOLL_INPUT|ABCDK_EPOLL_OUTPUT, ABCDK_EPOLL_INPUT) == 0);

                    break;
                }
            }
            if(e.events & ABCDK_EPOLL_OUTPUT)
            {
                while (1)
                {
                    static int fd = -1;
                    if(fd ==-1)
                        fd = abcdk_open("/var/log/syslog",0,0,0);
#if 0 
                    char buf[100];
                    size_t n = abcdk_read(fd, buf, 100);
                    if (n <= 0)
                    {
                        abcdk_closep(&fd);
                        assert(abcdk_mux_mark(m,e.data.fd, 0, ABCDK_EPOLL_OUTPUT) == 0);
                        break;
                    }

                    //memset(buf,rand()%26+40,100);
                    int s = send(e.data.fd, buf, 100, 0);
                    if (s > 0)
                        continue;
                    if (s == -1 && errno == EAGAIN)
                        assert(abcdk_mux_mark(m,e.data.fd, ABCDK_EPOLL_OUTPUT, ABCDK_EPOLL_OUTPUT) == 0);
#else
                    ssize_t s = sendfile(e.data.fd,fd,NULL,100000000);
                    printf("s=%ld\n",s);
                    if(s>0)
                        continue;
                    if (s == 0)
                    {
                        abcdk_closep(&fd);
                        assert(abcdk_mux_mark(m,e.data.fd, 0, ABCDK_EPOLL_OUTPUT) == 0);
                        break;
                    }
                    else if (s == -1 && errno == EAGAIN)
                    {
                        assert(abcdk_mux_mark(m,e.data.fd, ABCDK_EPOLL_OUTPUT, ABCDK_EPOLL_OUTPUT) == 0);
                    }
#endif
                    break;
                }
            }

            assert(abcdk_mux_unref(m,e.data.fd,e.events)==0);
        }
        
    }
}

void test_server(abcdk_tree_t *t)
{
    abcdk_clock_reset();

    abcdk_mux_t *m = abcdk_mux_alloc();
#if 0

    printf("attach begin:%lu\n",abcdk_clock_dot(NULL));

    for (int i = 0; i < 100000; i++)
        assert(abcdk_mux_attach(m, i, 100) == 0);
    //   assert(abcdk_mux_attach(m,10000,100)==0);

    printf("attach cast:%lu\n",abcdk_clock_step(NULL));

    getchar();
    
    printf("attach begin:%lu\n",abcdk_clock_dot(NULL));

    for (int i = 0; i < 100000; i++)
        assert(abcdk_mux_detach(m, i) == 0);

    printf("detach cast:%lu\n",abcdk_clock_step(NULL));
#else

    #pragma omp parallel for num_threads(3)
    for (int i = 0; i < 3; i++)
    {
#if 0
        abcdk_thread_t p;
        p.routine = server_loop;
        p.opaque = m;
        abcdk_thread_create(&p, 0);
#else 
        server_loop(m);
#endif 
    }

    while (getchar()!='Q');    

#endif

    abcdk_mux_free(&m);
}

void test_client(abcdk_tree_t *t)
{
    int c = abcdk_socket(ABCDK_IPV4, 0);
    abcdk_sockaddr_t a = {0};
    abcdk_sockaddr_from_string(&a, abcdk_option_get(t,"--",1,"localhost:12345"), 1);

    int chk = abcdk_connect(c,&a,10000);
    assert(chk==0);

    send(c,"aaa\n",3,0);

    while(1)
    {   
        char buf[100] ={0};
        int r = recv(c,buf,100,0);
        if(r<=0)
            break;
        printf("%s\n",buf);
    }

    abcdk_closep(&c);
}

void test_mux(abcdk_tree_t *args)
{
    if(abcdk_option_exist(args,"--server"))
        test_server(args);
    else 
        test_client(args);
}

int main(int argc, char **argv)
{
    abcdk_openlog(NULL,LOG_DEBUG,1);

    abcdk_thread_t p;
    p.routine = sigwaitinfo_cb;
    abcdk_thread_create(&p,0);

    abcdk_tree_t *args = abcdk_tree_alloc(NULL);

    abcdk_getargs(args,argc,argv,"--");

    const char *func = abcdk_option_get(args,"--func",0,"");

    if(abcdk_strcmp(func,"test_mux",0)==0)
        test_mux(args);

    abcdk_tree_free(&args);

    return 0;
}