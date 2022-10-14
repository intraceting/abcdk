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
#include "entry.h"

//environ

int abcdk_test_exec(abcdk_tree_t *args)
{
    int pid = abcdk_option_get_int(args,"--pid",0,-1);
    const char *cmd = abcdk_option_get(args,"--cmd",0,"/bin/bash");

    char *newargs[] = {"bash",NULL};
    char *newenvs[] = {NULL};

    
    char *nss[] = {"cgroup","ipc","mnt","pid","net","pid_for_children","user","uts"};

    for(int i = 0;i<8;i++)
    {
        char buf[100] = {0};
        sprintf(buf,"/proc/%d/ns/%s",pid,nss[i]);
        printf("%s\n",buf);
        int fd = abcdk_open(buf,0,0,0);
       // assert(setns(fd,0)==0);
        setns(fd,0);
        abcdk_closep(&fd);
    }
    
   // assert(unshare(CLONE_NEWIPC|CLONE_NEWNS|CLONE_NEWNET|CLONE_NEWPID|CLONE_NEWUTS|CLONE_NEWUSER)==0);

    assert(unshare(CLONE_NEWNET)==0);
    abcdk_exec(cmd,newargs,newenvs,0,0,NULL,NULL);

  //  execvp(cmd,&cmd);
}