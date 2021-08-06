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
#include <limits.h>
#include "abcdk-util/dirent.h"
#include "abcdk-util/mman.h"
#include "abcdk-util/notify.h"
#include "abcdk-util/iconv.h"
#include "abcdk-util/tar.h"
#include "abcdk-util/termios.h"

void test_dir()
{
    char* path = abcdk_heap_alloc(PATH_MAX);
    char* path2 = abcdk_heap_alloc(PATH_MAX);
    char* path3 = abcdk_heap_alloc(PATH_MAX);
    char* path4 = abcdk_heap_alloc(PATH_MAX);
    char* path5 = abcdk_heap_alloc(PATH_MAX);
    char* path6 = abcdk_heap_alloc(PATH_MAX);
    char* path7 = abcdk_heap_alloc(PATH_MAX);

    abcdk_dirdir(path,"/tmp/abcd/efg/heigh/././././///////sadfasdfasdfasfdsd/.a.a.sdf.../asdfasdfasfd/.../../../dfgsdfg///../..////////aaa.txt");

    abcdk_dirnice(path5,path);

    abcdk_mkdir(path5,0);

    abcdk_dirname(path2,path);

    abcdk_dirname(path6,path5);

    abcdk_basename(path3,path);

    printf("%s : %s\n",path2,path3);
    printf("%s : %s\n",path5,path6);

    abcdk_heap_free(path);
    abcdk_heap_free(path2);
    abcdk_heap_free(path3);
    abcdk_heap_free(path5);
    abcdk_heap_free(path6);

    strcpy(path4,"              asdfasdfsad     \r\n\t");

    printf("[%s]\n",path4);

    abcdk_strtrim(path4,iscntrl,2);
    abcdk_strtrim(path4,isblank,2);

    printf("[%s]\n",path4);

    //abcdk_abspath(path7, path4,NULL);
    //abcdk_abspath(path7, path4,"bbb");
    abcdk_abspath(path7, "/aaa","bbb");

    printf("[%s]\n",path7);

    abcdk_heap_free(path4);
    abcdk_heap_free(path7);


    
}

int compare_cb(const abcdk_tree_t *node1, const abcdk_tree_t *node2, void *opaque)
{
    char* src = ABCDK_PTR2PTR(char, node1->alloc->pptrs[ABCDK_DIRENT_NAME], 0);
    char* dst = ABCDK_PTR2PTR(char, node2->alloc->pptrs[ABCDK_DIRENT_NAME], 0);

    return abcdk_strcmp(src,dst,1);
}

void test_sort(abcdk_tree_t *t,int by)
{
    abcdk_tree_order_t o = {by,compare_cb,NULL};

    abcdk_tree_sort(t,&o);
}

int dump2(size_t deep, abcdk_tree_t *node, void *opaque)
{

    test_sort(node,0);

    char *path = (char*)(node->alloc->pptrs[ABCDK_DIRENT_NAME]);
    struct stat *stat = (struct stat *)(node->alloc->pptrs[ABCDK_DIRENT_STAT]);


    char name[NAME_MAX] ={0};
    abcdk_basename(name,path);

#if 1
    abcdk_tree_fprintf(stderr,deep,node,"%s%s\n",name,(S_ISDIR(stat->st_mode)?"/":""));
#else 
    abcdk_tree_fprintf(stderr,deep,node,"%s(%s)\n",name,path);
#endif
    return 1;
}

void traversal(abcdk_tree_t *root)
{
    printf("\n-------------------------------------\n");

    abcdk_tree_iterator_t it = {0,dump2,NULL};
    abcdk_tree_scan(root,&it);

    printf("\n-------------------------------------\n");
}

void test_dirscan()
{
    size_t sizes[2] = {PATH_MAX,sizeof(struct stat)};
    abcdk_tree_t * t = abcdk_tree_alloc2(sizes,2,0);

   strcpy(t->alloc->pptrs[ABCDK_DIRENT_NAME],"/tmp/");
   abcdk_dirscan(t,100,0);

//    strcpy(t->alloc->pptrs[ABCDK_DIRENT_NAME],"/proc/");
 //   abcdk_dirscan(t,5,0);

 //   strcpy(t->alloc->pptrs[ABCDK_DIRENT_NAME],"/usr");
 //   abcdk_dirscan(t,100,0);

 //   strcpy(t->alloc->pptrs[ABCDK_DIRENT_NAME],"/mnt");
 //   abcdk_dirscan(t,100,1);

 //   strcpy(t->alloc->pptrs[ABCDK_DIRENT_NAME],"/tmp");
  //  abcdk_dirscan(t,1,0);

    traversal(t);

    abcdk_tree_free(&t);
    
}

void test_file(const char *f1, const char *f2)
{
    abcdk_mkdir(f2, 0);

    int fd1 = abcdk_open(f1, 0, 1, 0);
    int fd2 = abcdk_open(f2, 1, 1, 1);

    assert(fd1 >= 0 && fd2 >= 0);

    abcdk_buffer_t *rbuf = abcdk_buffer_alloc2(512);
    abcdk_buffer_t *wbuf = abcdk_buffer_alloc2(256);

    while (1)
    {
        char buf[1023];
        ssize_t rsize = abcdk_block_read(fd1, buf, 1023, rbuf);
        if (rsize <= 0)
            break;

        ssize_t wsize = abcdk_block_write(fd2, buf, rsize, wbuf);

        assert(wsize == rsize);
    }

    assert(abcdk_block_write_trailer(fd2,0, wbuf) == 0);

    abcdk_buffer_free(&rbuf);
    abcdk_buffer_free(&wbuf);

    abcdk_closep(&fd1);
    abcdk_closep(&fd2);
}

void test_mman()
{
    abcdk_allocator_t * buf = abcdk_mmap2("/tmp/test_mman.txt",1,1);

    /*支持引用访问。*/
    abcdk_allocator_t * buf2 = abcdk_allocator_refer(buf);

    memset(buf->pptrs[0],'A',buf->sizes[0]);

    abcdk_munmap(&buf);

    assert(abcdk_msync(buf2,0)==0);

    abcdk_munmap(&buf2);


    /*如果映射的内存页面是私有模式，则对内存数据修改不会影响原文件。*/
    abcdk_allocator_t * buf3 = abcdk_mmap2("/tmp/test_mman.txt",1,0);

    memset(buf3->pptrs[0],'B',buf3->sizes[0]);

    assert(abcdk_msync(buf3,0)==0);

    abcdk_munmap(&buf3);

    /*创建共享内存文件。*/
    int fd = abcdk_shm_open("test_mman",1,1);

    /*划拔点内存空间。*/
    ftruncate(fd,100);

    abcdk_allocator_t * buf4 = abcdk_mmap(fd,1,1);

    memset(buf4->pptrs[0],'C',buf4->sizes[0]);

    abcdk_allocator_t * buf5  = abcdk_allocator_refer(buf4);

    abcdk_munmap(&buf4);

    assert(abcdk_msync(buf5,0)==0);

    abcdk_munmap(&buf5);

    abcdk_allocator_t * buf6 = abcdk_mmap(fd,1,0);

    printf("%s\n",buf6->pptrs[0]);

    abcdk_munmap(&buf6);

    abcdk_closep(&fd);

    /*删除*/
    abcdk_shm_unlink("test_mman");
}

void test_notify()
{
    int fd = abcdk_notify_init(1);

    abcdk_notify_event_t t = {0};

    t.buf = abcdk_buffer_alloc2(4096);

    //int wd = abcdk_notify_add(fd,"/tmp/",IN_ALL_EVENTS);
    int wd = abcdk_notify_add(fd,"/tmp/",IN_CREATE|IN_DELETE|IN_MOVE_SELF|IN_MOVE);

    for(;;)
    {
        

        if(abcdk_notify_watch(fd,&t,-1)<0)
            break;

        if(t.event.mask & IN_ACCESS )
            printf("Access:");
        if(t.event.mask & IN_MODIFY )
            printf("Modify:");
        if(t.event.mask & IN_ATTRIB )
            printf("Metadata changed:");
        if(t.event.mask & IN_CLOSE )
            printf("Close:");
        if(t.event.mask & IN_OPEN )
            printf("Open:");
        if(t.event.mask & IN_MOVED_FROM )
            printf("Moved from(%u):",t.event.cookie);
        if(t.event.mask & IN_MOVED_TO )
            printf("Moved to(%u):",t.event.cookie);
        if(t.event.mask & IN_CREATE )
            printf("Created:");
        if(t.event.mask & IN_DELETE )
            printf("Deleted:");
        if(t.event.mask & IN_MOVE_SELF )
            printf("Deleted self:");
        if(t.event.mask & IN_UNMOUNT )
            printf("Umount:");
        if(t.event.mask & IN_IGNORED )
            printf("Ignored:");

        printf("%s\n",t.name);
    }

    abcdk_buffer_free(&t.buf);

    abcdk_closep(&fd);
}

void test_iconv()
{
    char src[100] = {"我爱我家"};
    char dst[100] = {0};

    iconv_t cd = iconv_open("GBK","UTF8");

    size_t remain = 0;
    size_t n = abcdk_iconv(cd,src,strlen(src),dst,4,&remain);

    iconv_close(cd);
}

void test_tar_read(const char* tarfile)
{

    abcdk_tar_t t;

    t.fd = abcdk_open(tarfile,0,0,0);
    t.buf = abcdk_buffer_alloc2(512 * 1024);

    size_t bufsize = 1000000;
    void *buf=abcdk_heap_alloc(bufsize);

    while (1)
    {
        char name[PATH_MAX] = {0};
        struct stat attr = {0};
        char linkname[PATH_MAX] = {0};

        if (abcdk_tar_read_hdr(&t, name, &attr, linkname) != 0)
            break;

        printf("%s->%s\n", name, linkname);

        if(S_ISREG(attr.st_mode))
        {
            size_t size = 0;

            while (size<attr.st_size)
            {
                size_t rsize = abcdk_tar_read(&t,buf,ABCDK_MIN(attr.st_size-size,bufsize));
                assert(rsize>0);
                size += rsize;
            }
            
            if(attr.st_size>0)
                assert(abcdk_tar_read_align(&t,attr.st_size)==0);
        }

        
    }

    abcdk_heap_free2(&buf);
    abcdk_buffer_free(&t.buf);
    abcdk_closep(&t.fd);
}

int tar_dump(size_t depth, abcdk_tree_t *node, void *opaque)
{
    abcdk_allocator_t* m = NULL;
    abcdk_tar_t *t = (abcdk_tar_t*)opaque;

    if(depth == 0)
        return 1;


    char* name = node->alloc->pptrs[ABCDK_DIRENT_NAME];
    struct stat * attr = (struct stat*)node->alloc->pptrs[ABCDK_DIRENT_STAT];

    printf("%s\n",name);

    if(S_ISREG(attr->st_mode))
    {
        abcdk_tar_write_hdr(t,name,attr,NULL);

        if(attr->st_size>0)
        {
            m = abcdk_mmap2(name,0,0);
            if(!m)
                goto final_error;
            if(abcdk_tar_write(t,m->pptrs[0],m->sizes[0])!=m->sizes[0])
                goto final_error;
            
            if(abcdk_tar_write_align(t,m->sizes[0])!=0)
                goto final_error;

            abcdk_munmap(&m);
        }
    }
    else if(S_ISDIR(attr->st_mode))
        abcdk_tar_write_hdr(t,name,attr,NULL);
    else if(S_ISLNK(attr->st_mode))
    {
        char linkname[PATH_MAX] = {0};
        readlink(name,linkname,PATH_MAX);
        abcdk_tar_write_hdr(t,name,attr,linkname);
    }    

    return 1;

final_error:

     abcdk_munmap(&m);

    return -1;
}

void test_tar_write(const char* tarfile)
{
    abcdk_tar_t t;

    t.fd = abcdk_open(tarfile,1,0,1);
    t.buf = abcdk_buffer_alloc2(512 * 1024);

    size_t sizes[2] = {PATH_MAX,sizeof(struct stat)};
    abcdk_tree_t * dir = abcdk_tree_alloc2(sizes,2,0);

   strcpy(dir->alloc->pptrs[ABCDK_DIRENT_NAME],"/tmp/");
   abcdk_dirscan(dir,100,0);


    abcdk_tree_iterator_t it = {0,tar_dump,&t};
    abcdk_tree_scan(dir,&it);

    assert(abcdk_tar_write_trailer(&t,0)==0);
    
    abcdk_tree_free(&dir);
    abcdk_buffer_free(&t.buf);
    abcdk_closep(&t.fd);
}

void test_termios()
{
    printf("终端按键测试。\n");

    assert(abcdk_tcattr_cbreak(STDIN_FILENO,NULL)==0);
    
    while(1)
    {
        abcdk_poll(STDIN_FILENO,1,-1);

        char buf[10] = {0};

        long n = read(STDIN_FILENO,buf,10);

        for(long i = 0;i<n;i++)
            printf("%c(%02X)",buf[i],buf[i]);
        printf("\n");

    }
}

int main(int argc, char **argv)
{
    
     test_dir();

   //  test_dirscan();

 //    if(argc>=3)
  //       test_file(argv[1],argv[2]);

    // test_mman();

    //test_notify();

   // test_iconv();

 //   if(argc>=2)
  //       test_tar_read(argv[1]);

    
//    if(argc>=2)
//         test_tar_write(argv[1]);

  //  test_termios();

    return 0;
}