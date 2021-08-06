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
#include "abcdk-util/general.h"
#include "abcdk-util/scsi.h"
#include "abcdk-util/mtx.h"
#include "abcdk-util/mt.h"

void test_get_sn()
{
    int fd = abcdk_open("/dev/st0",0,0,0);

    abcdk_scsi_io_stat stat = {0};
    
    uint8_t type = 0;
    char vendor[32] = {0};
    char product[64] = {0};
    char sn[64]={0};

    assert(abcdk_scsi_inquiry_standard(fd,&type,vendor,product,3000,&stat)==0);

    printf("type:%s(%hhu),vendor:%s,product:%s",abcdk_scsi_type2string(type),type,vendor,product);

    assert(abcdk_scsi_inquiry_serial(fd,NULL,sn,3000,&stat)==0);
    printf(",sn:%s\n",sn);

    abcdk_closep(&fd);
}

int dump2(size_t deep, abcdk_tree_t *node, void *opaque)
{
    if(deep==0)
    {
        abcdk_tree_fprintf(stderr,deep,node,"haha\n");
    }
    else
    {
        abcdk_tree_fprintf(stderr, deep, node, "%-6hu\t|%-2hhu\t|%-2hhu\t|%-10s\t|%-10s\t|\n",
                          ABCDK_PTR2OBJ(uint16_t, node->alloc->pptrs[ABCDK_MTX_ELEMENT_ADDR], 0),
                          ABCDK_PTR2OBJ(uint8_t, node->alloc->pptrs[ABCDK_MTX_ELEMENT_TYPE], 0),
                          ABCDK_PTR2OBJ(uint8_t, node->alloc->pptrs[ABCDK_MTX_ELEMENT_ISFULL], 0),
                          node->alloc->pptrs[ABCDK_MTX_ELEMENT_BARCODE],
                          node->alloc->pptrs[ABCDK_MTX_ELEMENT_DVCID]);
    }

    return 1;
}

void traversal(abcdk_tree_t *root)
{
    printf("\n-------------------------------------\n");

    abcdk_tree_iterator_t it = {0,dump2,NULL};
    abcdk_tree_scan(root,&it);

    printf("\n-------------------------------------\n");
}


void test_mtx()
{
    int fd = abcdk_open("/dev/sg9",0,0,0);

    

    abcdk_scsi_io_stat stat = {0};

    assert(abcdk_mtx_inventory(fd,0,0,1000,&stat)==0);

    assert(abcdk_mtx_inventory(fd,5000,10,1000,&stat)==0);

    for (int i = 0; i < 4; i++)
    {
        assert(abcdk_mtx_move_medium(fd, 0,1029+i, 500+i, -1, &stat) == 0);

        printf("%hhx,%hhx,%hhx\n", abcdk_scsi_sense_key(stat.sense), abcdk_scsi_sense_code(stat.sense), abcdk_scsi_sense_qualifier(stat.sense));
    }

    // assert(abcdk_mtx_prevent_medium_removal(fd,1,-1,&stat)==0);

    //  printf("%hhx,%hhx,%hhx\n",abcdk_scsi_sense_key(stat.sense),abcdk_scsi_sense_code(stat.sense),abcdk_scsi_sense_qualifier(stat.sense));

    // assert(abcdk_mtx_prevent_medium_removal(fd,0,-1,&stat)==0);

    // printf("%hhx,%hhx,%hhx\n",abcdk_scsi_sense_key(stat.sense),abcdk_scsi_sense_code(stat.sense),abcdk_scsi_sense_qualifier(stat.sense));

    // char buf[255] = {0};
    // assert(abcdk_mtx_mode_sense(fd, 0, 0x1d, 0, buf, 255, -1, &stat) == 0);

    // /**/
    // uint16_t changer_address = abcdk_endian_b_to_h16(*ABCDK_PTR2PTR(uint16_t, buf, 4 + 2));
    // uint16_t changer_count = abcdk_endian_b_to_h16(*ABCDK_PTR2PTR(uint16_t, buf,4 + 4));
    // uint16_t storage_address = abcdk_endian_b_to_h16(*ABCDK_PTR2PTR(uint16_t, buf, 4 + 6));
    // uint16_t storage_count = abcdk_endian_b_to_h16(*ABCDK_PTR2PTR(uint16_t, buf, 4 + 8));
    // uint16_t storage_ie_address = abcdk_endian_b_to_h16(*ABCDK_PTR2PTR(uint16_t, buf, 4 + 10));
    // uint16_t storage_ie_count = abcdk_endian_b_to_h16(*ABCDK_PTR2PTR(uint16_t, buf, 4 + 12));
    // uint16_t driver_address = abcdk_endian_b_to_h16(*ABCDK_PTR2PTR(uint16_t, buf, 4 + 14));
    // uint16_t driver_count = abcdk_endian_b_to_h16(*ABCDK_PTR2PTR(uint16_t, buf, 4 + 16));

    // /**/
    // int buf2size = (0x00ffffff); /*15MB enough!*/
    // uint8_t * buf2 = (uint8_t*) abcdk_heap_alloc(buf2size);

    // assert(abcdk_mtx_read_element_status(fd,ABCDK_MXT_ELEMENT_DXFER,driver_address,driver_count,buf2,2*1024*1024,-1,&stat)==0);

    abcdk_tree_t *t = abcdk_tree_alloc(NULL);

    assert(abcdk_mtx_inquiry_element_status(t,fd,-1,&stat)==0);

    traversal(t);

    abcdk_tree_free(&t);

    abcdk_closep(&fd);
}

void test_mt()
{
    int fd = abcdk_open("/dev/st1",1,0,0);

    abcdk_scsi_io_stat stat = {0};

    abcdk_mt_compression(fd,0);
    abcdk_mt_blocksize(fd,0);

    assert(abcdk_mt_verify(fd,3000,&stat)==0);

    assert(abcdk_mt_locate(fd,0,0,100,3000,&stat)==0);

     printf("%hhx,%hhx,%hhx\n", abcdk_scsi_sense_key(stat.sense), abcdk_scsi_sense_code(stat.sense), abcdk_scsi_sense_qualifier(stat.sense));


    abcdk_write(fd,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",30);
    abcdk_write(fd,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",30);

  //  abcdk_mt_writefm(fd,10);

    uint64_t block = -1;
    uint64_t file = -1;
    uint32_t part = -1;

     assert(abcdk_mt_read_position(fd,&block,&file,&part,3000,&stat)==0);

     printf("%lu,%lu,%u\n",block,file,part);

     abcdk_allocator_t *a = abcdk_mt_read_attribute(fd,0,0x0000,100,&stat);
     abcdk_allocator_t *b = abcdk_mt_read_attribute(fd,0,0x0001,100,&stat);
     abcdk_allocator_t *c = abcdk_mt_read_attribute(fd,0,0x0400,100,&stat);
     abcdk_allocator_t *d = abcdk_mt_read_attribute(fd,0,0x0401,100,&stat);
     abcdk_allocator_t *e = abcdk_mt_read_attribute(fd,0,0x0405,100,&stat);
     abcdk_allocator_t *f = abcdk_mt_read_attribute(fd,0,0x0806,100,&stat);

     abcdk_endian_b_to_h(a->pptrs[ABCDK_MT_ATTR_VALUE],ABCDK_PTR2U16(a->pptrs[ABCDK_MT_ATTR_LENGTH],0));
     abcdk_endian_b_to_h(b->pptrs[ABCDK_MT_ATTR_VALUE],ABCDK_PTR2U16(b->pptrs[ABCDK_MT_ATTR_LENGTH],0));

     printf("REMAINING CAPACITY:%lu\n",ABCDK_PTR2U64(a->pptrs[ABCDK_MT_ATTR_VALUE], 0));
     printf("MAXIMUM CAPACITY:%lu\n",ABCDK_PTR2U64(b->pptrs[ABCDK_MT_ATTR_VALUE], 0));
     printf("MANUFACTURER:%s\n",c->pptrs[ABCDK_MT_ATTR_VALUE]);
     printf("SERIAL NUMBER:%s\n",d->pptrs[ABCDK_MT_ATTR_VALUE]);
     printf("DENSITY:%s\n",abcdk_mt_density2string(ABCDK_PTR2U8(e->pptrs[ABCDK_MT_ATTR_VALUE], 0)));
     printf("BARCODE:%s\n",f->pptrs[ABCDK_MT_ATTR_VALUE]);


    size_t sizes[5] = {sizeof(uint16_t), sizeof(uint8_t), sizeof(uint8_t), sizeof(uint16_t), 32+1};
    abcdk_allocator_t *g = abcdk_allocator_alloc(sizes,5,0);

    ABCDK_PTR2U16(g->pptrs[ABCDK_MT_ATTR_ID],0) = 0x0806;
    ABCDK_PTR2U16(g->pptrs[ABCDK_MT_ATTR_FORMAT],0) = 1;
    ABCDK_PTR2U16(g->pptrs[ABCDK_MT_ATTR_LENGTH],0) = 32;

    memcpy(g->pptrs[ABCDK_MT_ATTR_VALUE],"aaaaaa",5);

    assert(abcdk_mt_write_attribute(fd,0,g,3000,&stat)==0);


     abcdk_allocator_unref(&a);
     abcdk_allocator_unref(&b);
     abcdk_allocator_unref(&c);
     abcdk_allocator_unref(&d);
     abcdk_allocator_unref(&e);
     abcdk_allocator_unref(&f);
     abcdk_allocator_unref(&g);

    abcdk_closep(&fd);
}

int main(int argc, char **argv)
{

    test_get_sn();

    test_mtx();

    test_mt();

    return 0;
}

