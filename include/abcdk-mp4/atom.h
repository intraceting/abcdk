/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_MP4_MP4_H
#define ABCDK_MP4_MP4_H

#include "abcdk-util/general.h"
#include "abcdk-util/allocator.h"

__BEGIN_DECLS

/*
 * tag构造宏(大端字节序)。
*/

# if __BYTE_ORDER == __LITTLE_ENDIAN
    #define ABCDK_MP4_ATOM_MKTAG(a, b, c, d) ((a) | ((b) << 8) | ((c) << 16) | ((uint32_t)(d) << 24))
#else 
    #define ABCDK_MP4_ATOM_MKTAG(a, b, c, d) ((d) | ((c) << 8) | ((b) << 16) | ((uint32_t)(a) << 24))
#endif

/** MP4原子结构。*/
typedef struct _abcdk_mp4_atom
{
    /**
     * 长度(字节，头部+内容)。
    */
    uint64_t size;

    /**
     * 类型。
     * 
     * @note 大端字节序。
    */
    union
    {
        /** char*/
        uint8_t u8[4];

        /** int*/
        uint32_t u32;
    } type;

    /** 
     * 头部偏移量(字节)。
     * 
     * -demuxer: 有效。
     * -muxer: 无效。
    */
    uint64_t off_head;

    /** 
     * 内容偏移量(字节)。
     * 
     * -demuxer: 有效。
     * -muxer: 无效。
    */
    uint64_t off_cont;

    /**
     * 数据体(头部+内容)。
    */
    abcdk_allocator_t *data;

} abcdk_mp4_atom_t;


/*
 * atom types
 * 
 * @note 从Bento4复制来的。
*/

#define ABCDK_MP4_ATOM_TYPE_UDTA ABCDK_MP4_ATOM_MKTAG('u', 'd', 't', 'a')
#define ABCDK_MP4_ATOM_TYPE_URL ABCDK_MP4_ATOM_MKTAG('u', 'r', 'l', ' ')
#define ABCDK_MP4_ATOM_TYPE_TRAK ABCDK_MP4_ATOM_MKTAG('t', 'r', 'a', 'k')
#define ABCDK_MP4_ATOM_TYPE_TRAF ABCDK_MP4_ATOM_MKTAG('t', 'r', 'a', 'f')
#define ABCDK_MP4_ATOM_TYPE_TKHD ABCDK_MP4_ATOM_MKTAG('t', 'k', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_TFHD ABCDK_MP4_ATOM_MKTAG('t', 'f', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_TRUN ABCDK_MP4_ATOM_MKTAG('t', 'r', 'u', 'n')
#define ABCDK_MP4_ATOM_TYPE_STTS ABCDK_MP4_ATOM_MKTAG('s', 't', 't', 's')
#define ABCDK_MP4_ATOM_TYPE_STSZ ABCDK_MP4_ATOM_MKTAG('s', 't', 's', 'z')
#define ABCDK_MP4_ATOM_TYPE_STZ2 ABCDK_MP4_ATOM_MKTAG('s', 't', 'z', '2')
#define ABCDK_MP4_ATOM_TYPE_STSS ABCDK_MP4_ATOM_MKTAG('s', 't', 's', 's')
#define ABCDK_MP4_ATOM_TYPE_STSD ABCDK_MP4_ATOM_MKTAG('s', 't', 's', 'd')
#define ABCDK_MP4_ATOM_TYPE_STSC ABCDK_MP4_ATOM_MKTAG('s', 't', 's', 'c')
#define ABCDK_MP4_ATOM_TYPE_STCO ABCDK_MP4_ATOM_MKTAG('s', 't', 'c', 'o')
#define ABCDK_MP4_ATOM_TYPE_CO64 ABCDK_MP4_ATOM_MKTAG('c', 'o', '6', '4')
#define ABCDK_MP4_ATOM_TYPE_STBL ABCDK_MP4_ATOM_MKTAG('s', 't', 'b', 'l')
#define ABCDK_MP4_ATOM_TYPE_SINF ABCDK_MP4_ATOM_MKTAG('s', 'i', 'n', 'f')
#define ABCDK_MP4_ATOM_TYPE_SCHM ABCDK_MP4_ATOM_MKTAG('s', 'c', 'h', 'm')
#define ABCDK_MP4_ATOM_TYPE_SCHI ABCDK_MP4_ATOM_MKTAG('s', 'c', 'h', 'i')
#define ABCDK_MP4_ATOM_TYPE_MVHD ABCDK_MP4_ATOM_MKTAG('m', 'v', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_MEHD ABCDK_MP4_ATOM_MKTAG('m', 'e', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_MP4S ABCDK_MP4_ATOM_MKTAG('m', 'p', '4', 's')
#define ABCDK_MP4_ATOM_TYPE_MP4A ABCDK_MP4_ATOM_MKTAG('m', 'p', '4', 'a')
#define ABCDK_MP4_ATOM_TYPE_MP4V ABCDK_MP4_ATOM_MKTAG('m', 'p', '4', 'v')
#define ABCDK_MP4_ATOM_TYPE_AVC1 ABCDK_MP4_ATOM_MKTAG('a', 'v', 'c', '1')
#define ABCDK_MP4_ATOM_TYPE_AVC2 ABCDK_MP4_ATOM_MKTAG('a', 'v', 'c', '2')
#define ABCDK_MP4_ATOM_TYPE_AVC3 ABCDK_MP4_ATOM_MKTAG('a', 'v', 'c', '3')
#define ABCDK_MP4_ATOM_TYPE_AVC4 ABCDK_MP4_ATOM_MKTAG('a', 'v', 'c', '4')
#define ABCDK_MP4_ATOM_TYPE_DVAV ABCDK_MP4_ATOM_MKTAG('d', 'v', 'a', 'v')
#define ABCDK_MP4_ATOM_TYPE_DVA1 ABCDK_MP4_ATOM_MKTAG('d', 'v', 'a', '1')
#define ABCDK_MP4_ATOM_TYPE_HEV1 ABCDK_MP4_ATOM_MKTAG('h', 'e', 'v', '1')
#define ABCDK_MP4_ATOM_TYPE_HVC1 ABCDK_MP4_ATOM_MKTAG('h', 'v', 'c', '1')
#define ABCDK_MP4_ATOM_TYPE_DVHE ABCDK_MP4_ATOM_MKTAG('d', 'v', 'h', 'e')
#define ABCDK_MP4_ATOM_TYPE_DVH1 ABCDK_MP4_ATOM_MKTAG('d', 'v', 'h', '1')
#define ABCDK_MP4_ATOM_TYPE_VP08 ABCDK_MP4_ATOM_MKTAG('v', 'p', '0', '8')
#define ABCDK_MP4_ATOM_TYPE_VP09 ABCDK_MP4_ATOM_MKTAG('v', 'p', '0', '9')
#define ABCDK_MP4_ATOM_TYPE_VP10 ABCDK_MP4_ATOM_MKTAG('v', 'p', '1', '0')
#define ABCDK_MP4_ATOM_TYPE_AV01 ABCDK_MP4_ATOM_MKTAG('a', 'v', '0', '1')
#define ABCDK_MP4_ATOM_TYPE_ALAC ABCDK_MP4_ATOM_MKTAG('a', 'l', 'a', 'c')
#define ABCDK_MP4_ATOM_TYPE_ENCA ABCDK_MP4_ATOM_MKTAG('e', 'n', 'c', 'a')
#define ABCDK_MP4_ATOM_TYPE_ENCV ABCDK_MP4_ATOM_MKTAG('e', 'n', 'c', 'v')
#define ABCDK_MP4_ATOM_TYPE_MOOV ABCDK_MP4_ATOM_MKTAG('m', 'o', 'o', 'v')
#define ABCDK_MP4_ATOM_TYPE_MOOF ABCDK_MP4_ATOM_MKTAG('m', 'o', 'o', 'f')
#define ABCDK_MP4_ATOM_TYPE_MVEX ABCDK_MP4_ATOM_MKTAG('m', 'v', 'e', 'x')
#define ABCDK_MP4_ATOM_TYPE_TREX ABCDK_MP4_ATOM_MKTAG('t', 'r', 'e', 'x')
#define ABCDK_MP4_ATOM_TYPE_MINF ABCDK_MP4_ATOM_MKTAG('m', 'i', 'n', 'f')
#define ABCDK_MP4_ATOM_TYPE_META ABCDK_MP4_ATOM_MKTAG('m', 'e', 't', 'a')
#define ABCDK_MP4_ATOM_TYPE_MDHD ABCDK_MP4_ATOM_MKTAG('m', 'd', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_MFHD ABCDK_MP4_ATOM_MKTAG('m', 'f', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_ILST ABCDK_MP4_ATOM_MKTAG('i', 'l', 's', 't')
#define ABCDK_MP4_ATOM_TYPE_HDLR ABCDK_MP4_ATOM_MKTAG('h', 'd', 'l', 'r')
#define ABCDK_MP4_ATOM_TYPE_FTYP ABCDK_MP4_ATOM_MKTAG('f', 't', 'y', 'p')
#define ABCDK_MP4_ATOM_TYPE_IODS ABCDK_MP4_ATOM_MKTAG('i', 'o', 'd', 's')
#define ABCDK_MP4_ATOM_TYPE_ESDS ABCDK_MP4_ATOM_MKTAG('e', 's', 'd', 's')
#define ABCDK_MP4_ATOM_TYPE_EDTS ABCDK_MP4_ATOM_MKTAG('e', 'd', 't', 's')
#define ABCDK_MP4_ATOM_TYPE_DRMS ABCDK_MP4_ATOM_MKTAG('d', 'r', 'm', 's')
#define ABCDK_MP4_ATOM_TYPE_DRMI ABCDK_MP4_ATOM_MKTAG('d', 'r', 'm', 'i')
#define ABCDK_MP4_ATOM_TYPE_DREF ABCDK_MP4_ATOM_MKTAG('d', 'r', 'e', 'f')
#define ABCDK_MP4_ATOM_TYPE_DINF ABCDK_MP4_ATOM_MKTAG('d', 'i', 'n', 'f')
#define ABCDK_MP4_ATOM_TYPE_CTTS ABCDK_MP4_ATOM_MKTAG('c', 't', 't', 's')
#define ABCDK_MP4_ATOM_TYPE_MDIA ABCDK_MP4_ATOM_MKTAG('m', 'd', 'i', 'a')
#define ABCDK_MP4_ATOM_TYPE_ELST ABCDK_MP4_ATOM_MKTAG('e', 'l', 's', 't')
#define ABCDK_MP4_ATOM_TYPE_VMHD ABCDK_MP4_ATOM_MKTAG('v', 'm', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_SMHD ABCDK_MP4_ATOM_MKTAG('s', 'm', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_NMHD ABCDK_MP4_ATOM_MKTAG('n', 'm', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_STHD ABCDK_MP4_ATOM_MKTAG('s', 't', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_HMHD ABCDK_MP4_ATOM_MKTAG('h', 'm', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_FRMA ABCDK_MP4_ATOM_MKTAG('f', 'r', 'm', 'a')
#define ABCDK_MP4_ATOM_TYPE_MDAT ABCDK_MP4_ATOM_MKTAG('m', 'd', 'a', 't')
#define ABCDK_MP4_ATOM_TYPE_FREE ABCDK_MP4_ATOM_MKTAG('f', 'r', 'e', 'e')
#define ABCDK_MP4_ATOM_TYPE_TIMS ABCDK_MP4_ATOM_MKTAG('t', 'i', 'm', 's')
#define ABCDK_MP4_ATOM_TYPE_RTP_ ABCDK_MP4_ATOM_MKTAG('r', 't', 'p', ' ')
#define ABCDK_MP4_ATOM_TYPE_HNTI ABCDK_MP4_ATOM_MKTAG('h', 'n', 't', 'i')
#define ABCDK_MP4_ATOM_TYPE_SDP_ ABCDK_MP4_ATOM_MKTAG('s', 'd', 'p', ' ')
#define ABCDK_MP4_ATOM_TYPE_IKMS ABCDK_MP4_ATOM_MKTAG('i', 'K', 'M', 'S')
#define ABCDK_MP4_ATOM_TYPE_ISFM ABCDK_MP4_ATOM_MKTAG('i', 'S', 'F', 'M')
#define ABCDK_MP4_ATOM_TYPE_ISLT ABCDK_MP4_ATOM_MKTAG('i', 'S', 'L', 'T')
#define ABCDK_MP4_ATOM_TYPE_TREF ABCDK_MP4_ATOM_MKTAG('t', 'r', 'e', 'f')
#define ABCDK_MP4_ATOM_TYPE_HINT ABCDK_MP4_ATOM_MKTAG('h', 'i', 'n', 't')
#define ABCDK_MP4_ATOM_TYPE_CDSC ABCDK_MP4_ATOM_MKTAG('c', 'd', 's', 'c')
#define ABCDK_MP4_ATOM_TYPE_MPOD ABCDK_MP4_ATOM_MKTAG('m', 'p', 'o', 'd')
#define ABCDK_MP4_ATOM_TYPE_IPIR ABCDK_MP4_ATOM_MKTAG('i', 'p', 'i', 'r')
#define ABCDK_MP4_ATOM_TYPE_CHAP ABCDK_MP4_ATOM_MKTAG('c', 'h', 'a', 'p')
#define ABCDK_MP4_ATOM_TYPE_ALIS ABCDK_MP4_ATOM_MKTAG('a', 'l', 'i', 's')
#define ABCDK_MP4_ATOM_TYPE_SYNC ABCDK_MP4_ATOM_MKTAG('s', 'y', 'n', 'c')
#define ABCDK_MP4_ATOM_TYPE_DPND ABCDK_MP4_ATOM_MKTAG('d', 'p', 'n', 'd')
#define ABCDK_MP4_ATOM_TYPE_ODRM ABCDK_MP4_ATOM_MKTAG('o', 'd', 'r', 'm')
#define ABCDK_MP4_ATOM_TYPE_ODKM ABCDK_MP4_ATOM_MKTAG('o', 'd', 'k', 'm')
#define ABCDK_MP4_ATOM_TYPE_OHDR ABCDK_MP4_ATOM_MKTAG('o', 'h', 'd', 'r')
#define ABCDK_MP4_ATOM_TYPE_ODDA ABCDK_MP4_ATOM_MKTAG('o', 'd', 'd', 'a')
#define ABCDK_MP4_ATOM_TYPE_ODHE ABCDK_MP4_ATOM_MKTAG('o', 'd', 'h', 'e')
#define ABCDK_MP4_ATOM_TYPE_ODAF ABCDK_MP4_ATOM_MKTAG('o', 'd', 'a', 'f')
#define ABCDK_MP4_ATOM_TYPE_GRPI ABCDK_MP4_ATOM_MKTAG('g', 'r', 'p', 'i')
#define ABCDK_MP4_ATOM_TYPE_IPRO ABCDK_MP4_ATOM_MKTAG('i', 'p', 'r', 'o')
#define ABCDK_MP4_ATOM_TYPE_MDRI ABCDK_MP4_ATOM_MKTAG('m', 'd', 'r', 'i')
#define ABCDK_MP4_ATOM_TYPE_AVCC ABCDK_MP4_ATOM_MKTAG('a', 'v', 'c', 'C')
#define ABCDK_MP4_ATOM_TYPE_HVCC ABCDK_MP4_ATOM_MKTAG('h', 'v', 'c', 'C')
#define ABCDK_MP4_ATOM_TYPE_DVCC ABCDK_MP4_ATOM_MKTAG('d', 'v', 'c', 'C')
#define ABCDK_MP4_ATOM_TYPE_VPCC ABCDK_MP4_ATOM_MKTAG('v', 'p', 'c', 'C')
#define ABCDK_MP4_ATOM_TYPE_DVVC ABCDK_MP4_ATOM_MKTAG('d', 'v', 'v', 'C')
#define ABCDK_MP4_ATOM_TYPE_HVCE ABCDK_MP4_ATOM_MKTAG('h', 'v', 'c', 'E')
#define ABCDK_MP4_ATOM_TYPE_AVCE ABCDK_MP4_ATOM_MKTAG('a', 'v', 'c', 'E')
#define ABCDK_MP4_ATOM_TYPE_AV1C ABCDK_MP4_ATOM_MKTAG('a', 'v', '1', 'C')
#define ABCDK_MP4_ATOM_TYPE_WAVE ABCDK_MP4_ATOM_MKTAG('w', 'a', 'v', 'e')
#define ABCDK_MP4_ATOM_TYPE_WIDE ABCDK_MP4_ATOM_MKTAG('w', 'i', 'd', 'e')
#define ABCDK_MP4_ATOM_TYPE_UUID ABCDK_MP4_ATOM_MKTAG('u', 'u', 'i', 'd')
#define ABCDK_MP4_ATOM_TYPE_8ID_ ABCDK_MP4_ATOM_MKTAG('8', 'i', 'd', ' ')
#define ABCDK_MP4_ATOM_TYPE_8BDL ABCDK_MP4_ATOM_MKTAG('8', 'b', 'd', 'l')
#define ABCDK_MP4_ATOM_TYPE_AC_3 ABCDK_MP4_ATOM_MKTAG('a', 'c', '-', '3')
#define ABCDK_MP4_ATOM_TYPE_EC_3 ABCDK_MP4_ATOM_MKTAG('e', 'c', '-', '3')
#define ABCDK_MP4_ATOM_TYPE_AC_4 ABCDK_MP4_ATOM_MKTAG('a', 'c', '-', '4')
#define ABCDK_MP4_ATOM_TYPE_DTSC ABCDK_MP4_ATOM_MKTAG('d', 't', 's', 'c')
#define ABCDK_MP4_ATOM_TYPE_DTSH ABCDK_MP4_ATOM_MKTAG('d', 't', 's', 'h')
#define ABCDK_MP4_ATOM_TYPE_DTSL ABCDK_MP4_ATOM_MKTAG('d', 't', 's', 'l')
#define ABCDK_MP4_ATOM_TYPE_DTSE ABCDK_MP4_ATOM_MKTAG('d', 't', 's', 'e')
#define ABCDK_MP4_ATOM_TYPE_FLAC ABCDK_MP4_ATOM_MKTAG('f', 'L', 'a', 'C')
#define ABCDK_MP4_ATOM_TYPE_OPUS ABCDK_MP4_ATOM_MKTAG('O', 'p', 'u', 's')
#define ABCDK_MP4_ATOM_TYPE_MFRA ABCDK_MP4_ATOM_MKTAG('m', 'f', 'r', 'a')
#define ABCDK_MP4_ATOM_TYPE_TFRA ABCDK_MP4_ATOM_MKTAG('t', 'f', 'r', 'a')
#define ABCDK_MP4_ATOM_TYPE_MFRO ABCDK_MP4_ATOM_MKTAG('m', 'f', 'r', 'o')
#define ABCDK_MP4_ATOM_TYPE_TFDT ABCDK_MP4_ATOM_MKTAG('t', 'f', 'd', 't')
#define ABCDK_MP4_ATOM_TYPE_TENC ABCDK_MP4_ATOM_MKTAG('t', 'e', 'n', 'c')
#define ABCDK_MP4_ATOM_TYPE_SENC ABCDK_MP4_ATOM_MKTAG('s', 'e', 'n', 'c')
#define ABCDK_MP4_ATOM_TYPE_SAIO ABCDK_MP4_ATOM_MKTAG('s', 'a', 'i', 'o')
#define ABCDK_MP4_ATOM_TYPE_SAIZ ABCDK_MP4_ATOM_MKTAG('s', 'a', 'i', 'z')
#define ABCDK_MP4_ATOM_TYPE_PDIN ABCDK_MP4_ATOM_MKTAG('p', 'd', 'i', 'n')
#define ABCDK_MP4_ATOM_TYPE_BLOC ABCDK_MP4_ATOM_MKTAG('b', 'l', 'o', 'c')
#define ABCDK_MP4_ATOM_TYPE_AINF ABCDK_MP4_ATOM_MKTAG('a', 'i', 'n', 'f')
#define ABCDK_MP4_ATOM_TYPE_PSSH ABCDK_MP4_ATOM_MKTAG('p', 's', 's', 'h')
#define ABCDK_MP4_ATOM_TYPE_MARL ABCDK_MP4_ATOM_MKTAG('m', 'a', 'r', 'l')
#define ABCDK_MP4_ATOM_TYPE_MKID ABCDK_MP4_ATOM_MKTAG('m', 'k', 'i', 'd')
#define ABCDK_MP4_ATOM_TYPE_PRFT ABCDK_MP4_ATOM_MKTAG('p', 'r', 'f', 't')
#define ABCDK_MP4_ATOM_TYPE_STPP ABCDK_MP4_ATOM_MKTAG('s', 't', 'p', 'p')
#define ABCDK_MP4_ATOM_TYPE_DAC3 ABCDK_MP4_ATOM_MKTAG('d', 'a', 'c', '3')
#define ABCDK_MP4_ATOM_TYPE_DEC3 ABCDK_MP4_ATOM_MKTAG('d', 'e', 'c', '3')
#define ABCDK_MP4_ATOM_TYPE_DAC4 ABCDK_MP4_ATOM_MKTAG('d', 'a', 'c', '4')
#define ABCDK_MP4_ATOM_TYPE_SIDX ABCDK_MP4_ATOM_MKTAG('s', 'i', 'd', 'x')
#define ABCDK_MP4_ATOM_TYPE_SSIX ABCDK_MP4_ATOM_MKTAG('s', 's', 'i', 'x')
#define ABCDK_MP4_ATOM_TYPE_SBGP ABCDK_MP4_ATOM_MKTAG('s', 'b', 'g', 'p')
#define ABCDK_MP4_ATOM_TYPE_SGPD ABCDK_MP4_ATOM_MKTAG('s', 'g', 'p', 'd')
#define ABCDK_MP4_ATOM_TYPE_SKIP ABCDK_MP4_ATOM_MKTAG('s', 'k', 'i', 'p')
#define ABCDK_MP4_ATOM_TYPE_IPMC ABCDK_MP4_ATOM_MKTAG('i', 'p', 'm', 'c')




/*
 * file type/brands
 *
 * @note 从Bento4复制来的。
*/

#define ABCDK_MP4_ATOM_FTYP_BRAND_QT__ ABCDK_MP4_ATOM_MKTAG('q', 't', ' ', ' ')
#define ABCDK_MP4_ATOM_FTYP_BRAND_ISOM ABCDK_MP4_ATOM_MKTAG('i', 's', 'o', 'm')
#define ABCDK_MP4_ATOM_FTYP_BRAND_ISO1 ABCDK_MP4_ATOM_MKTAG('i', 's', 'o', '1')
#define ABCDK_MP4_ATOM_FTYP_BRAND_ISO2 ABCDK_MP4_ATOM_MKTAG('i', 's', 'o', '2')
#define ABCDK_MP4_ATOM_FTYP_BRAND_ISO3 ABCDK_MP4_ATOM_MKTAG('i', 's', 'o', '3')
#define ABCDK_MP4_ATOM_FTYP_BRAND_ISO4 ABCDK_MP4_ATOM_MKTAG('i', 's', 'o', '4')
#define ABCDK_MP4_ATOM_FTYP_BRAND_ISO5 ABCDK_MP4_ATOM_MKTAG('i', 's', 'o', '5')
#define ABCDK_MP4_ATOM_FTYP_BRAND_ISO6 ABCDK_MP4_ATOM_MKTAG('i', 's', 'o', '6')
#define ABCDK_MP4_ATOM_FTYP_BRAND_MP41 ABCDK_MP4_ATOM_MKTAG('m', 'p', '4', '1')
#define ABCDK_MP4_ATOM_FTYP_BRAND_MP42 ABCDK_MP4_ATOM_MKTAG('m', 'p', '4', '2')
#define ABCDK_MP4_ATOM_FTYP_BRAND_3GP1 ABCDK_MP4_ATOM_MKTAG('3', 'g', 'p', '1')
#define ABCDK_MP4_ATOM_FTYP_BRAND_3GP2 ABCDK_MP4_ATOM_MKTAG('3', 'g', 'p', '2')
#define ABCDK_MP4_ATOM_FTYP_BRAND_3GP3 ABCDK_MP4_ATOM_MKTAG('3', 'g', 'p', '3')
#define ABCDK_MP4_ATOM_FTYP_BRAND_3GP4 ABCDK_MP4_ATOM_MKTAG('3', 'g', 'p', '4')
#define ABCDK_MP4_ATOM_FTYP_BRAND_3GP5 ABCDK_MP4_ATOM_MKTAG('3', 'g', 'p', '5')
#define ABCDK_MP4_ATOM_FTYP_BRAND_3G2A ABCDK_MP4_ATOM_MKTAG('3', 'g', '2', 'a')
#define ABCDK_MP4_ATOM_FTYP_BRAND_MMP4 ABCDK_MP4_ATOM_MKTAG('m', 'm', 'p', '4')
#define ABCDK_MP4_ATOM_FTYP_BRAND_M4A_ ABCDK_MP4_ATOM_MKTAG('M', '4', 'A', ' ')
#define ABCDK_MP4_ATOM_FTYP_BRAND_M4P_ ABCDK_MP4_ATOM_MKTAG('M', '4', 'P', ' ')
#define ABCDK_MP4_ATOM_FTYP_BRAND_MJP2 ABCDK_MP4_ATOM_MKTAG('m', 'j', 'p', '2')
#define ABCDK_MP4_ATOM_FTYP_BRAND_ODCF ABCDK_MP4_ATOM_MKTAG('o', 'd', 'c', 'f')
#define ABCDK_MP4_ATOM_FTYP_BRAND_OPF2 ABCDK_MP4_ATOM_MKTAG('o', 'p', 'f', '2')
#define ABCDK_MP4_ATOM_FTYP_BRAND_AVC1 ABCDK_MP4_ATOM_MKTAG('a', 'v', 'c', '1')
#define ABCDK_MP4_ATOM_FTYP_BRAND_HVC1 ABCDK_MP4_ATOM_MKTAG('h', 'v', 'c', '1')
#define ABCDK_MP4_ATOM_FTYP_BRAND_DBY1 ABCDK_MP4_ATOM_MKTAG('d', 'b', 'y', '1')


__END_DECLS

#endif //ABCDK_MP4_MP4_H