/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_MP4_ATOM_H
#define ABCDK_MP4_ATOM_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/time.h"
#include "abcdk/util/endian.h"

__BEGIN_DECLS


/** MP4 tag.*/
typedef union _abcdk_mp4_tag
{
    /** 字符型.*/
    uint8_t u8[4];

    /** 
     * 整型.
     * 
     * @note 大端字节序.
    */
    uint32_t u32;

}abcdk_mp4_tag_t;

/** MP4 tag 构造宏.*/
#define ABCDK_MP4_ATOM_MKTAG    ABCDK_FOURCC_MKTAG

/*
 * atom types
 * 
 * @note 从Bento4复制来的.
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
#define ABCDK_MP4_ATOM_TYPE_GMHD ABCDK_MP4_ATOM_MKTAG('g', 'm', 'h', 'd')
#define ABCDK_MP4_ATOM_TYPE_GLBL ABCDK_MP4_ATOM_MKTAG('g', 'l', 'b', 'l')
#define ABCDK_MP4_ATOM_TYPE_PRIV ABCDK_MP4_ATOM_MKTAG('p', 'r', 'i', 'v')


/*
 * file type/brands
 *
 * @note 从Bento4复制来的.
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

/** MP4 ftyp atom.*/
typedef struct _abcdk_mp4_atom_ftyp
{
    /** 主版本.*/
    abcdk_mp4_tag_t major;

    /** 副版本.*/
    uint32_t minor;

    /** 兼容版本.*/
    abcdk_object_t *compat;

}abcdk_mp4_atom_ftyp_t;


/** MP4 mvhd atom.*/
typedef struct _abcdk_mp4_atom_mvhd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 
     * 创建时间(秒), 开始于1904-01-01 00:00:00 +0000 (UTC).
     * 
     * 1970 to 1904 : +2082844800(0x7C25B080)
    */
    uint64_t ctime;

    /** 
     * 修改时间(秒), 开始于1904-01-01 00:00:00 +0000 (UTC).
     * 
     * 1970 to 1904 : +2082844800(0x7C25B080)
    */
    uint64_t mtime;

    /** 时间的刻度值(可以理解为1秒被分成多少份).*/
    uint32_t timescale;

    /** 时长(秒×时间的刻度值).*/
    uint64_t duration;

    /** 
     * 推荐播放速率(以整数形式存储的定点数).
     * 
     * 高16位: 整数.
     * 低16位: 小数. 
    */
    uint32_t rate;

    /** 
     * 推荐音量(以整数形式存储的定点数).
     * 
     * 高8位: 整数.
     * 低8位: 小数. 
    */
    uint16_t volume;

    /** 预留的.*/
    uint8_t reserved[10];

    /**变换矩阵.*/
    uint32_t matrix[9];

    /** 预定义.*/
    uint32_t predefined[6];

    /** 下一个TRACK ID.*/
    uint32_t nexttrackid;

}abcdk_mp4_atom_mvhd_t;


/** MP4 tkhd atom.*/
typedef struct _abcdk_mp4_atom_tkhd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 
     * 创建时间(秒), 开始于1904-01-01 00:00:00 +0000 (UTC).
     * 
     * 1970 to 1904 : +2082844800(0x7C25B080)
    */
    uint64_t ctime;

    /** 
     * 修改时间(秒), 开始于1904-01-01 00:00:00 +0000 (UTC).
     * 
     * 1970 to 1904 : +2082844800(0x7C25B080)
    */
    uint64_t mtime;

    /** TRACK ID.*/
    uint32_t trackid;

    /** 预留.*/
    uint32_t reserved1;

    /** 时长(秒×时间的刻度值).*/
    uint64_t duration;

    /** 预留.*/
    uint64_t reserved2;

    /** 预留.*/
    uint16_t layer;

    /** 预留.*/
    uint16_t alternategroup;

    /** 
     * 音量(以整数形式存储的定点数).
     * 
     * 高8位: 整数.
     * 低8位: 小数. 
     * 
     * -video: 无效.
     * -sound: 有效.
    */
    uint16_t volume;

    /** 预留.*/
    uint16_t reserved3;

    /** 变换矩阵.*/
    uint32_t matrix[9];

    /** 
     * 宽(以整数形式存储的定点数)(像素).
     * 
     * 高16位: 整数.
     * 低16位: 小数. 
    */
    uint32_t width;

    /** 
     * 高(以整数形式存储的定点数)(像素).
     * 
     * 高16位: 整数.
     * 低16位: 小数. 
    */
    uint32_t height;

}abcdk_mp4_atom_tkhd_t;

/** MP4 mdhd atom.*/
typedef struct _abcdk_mp4_atom_mdhd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 
     * 创建时间(秒), 开始于1904-01-01 00:00:00 +0000 (UTC).
     * 
     * 1970 to 1904 : +2082844800(0x7C25B080)
    */
    uint64_t ctime;

    /** 
     * 修改时间(秒), 开始于1904-01-01 00:00:00 +0000 (UTC).
     * 
     * 1970 to 1904 : +2082844800(0x7C25B080)
    */
    uint64_t mtime;

    /** 时间的刻度值(可以理解为1秒被分成多少份).*/
    uint32_t timescale;

    /** 时长(秒×时间的刻度值).*/
    uint64_t duration;

    /** 语言.*/
    uint16_t language;

    /** 质量.*/
    uint16_t quality;

}abcdk_mp4_atom_mdhd_t;

/** MP4 hdlr atom.*/
typedef struct _abcdk_mp4_atom_hdlr
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;
    
    /** 类型.*/
    abcdk_mp4_tag_t type;

    /** 子类型.*/
    abcdk_mp4_tag_t subtype;

    /** 预留.*/
    uint32_t reserved[3];

    /** 名称.*/
    abcdk_object_t *name;

}abcdk_mp4_atom_hdlr_t;


/** MP4 vmhd atom.*/
typedef struct _abcdk_mp4_atom_vmhd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 图像模式.*/
    uint16_t mode;

    /** 
     * 操作颜色.
     * 
     * [0]: red
     * [1]: green
     * [2]: blue
    */
    uint16_t opcolor[3];

}abcdk_mp4_atom_vmhd_t;


/** MP4 dref atom.*/
typedef struct _abcdk_mp4_atom_dref
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 子项数量*/
    uint32_t numbers;

}abcdk_mp4_atom_dref_t;

/** MP4 stsd atom.*/
typedef struct _abcdk_mp4_atom_stsd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 子项数量*/
    uint32_t numbers;

}abcdk_mp4_atom_stsd_t;

/** MP4 sample description atom.*/
typedef struct _abcdk_mp4_atom_sample_desc
{   
    /** 预留.*/
    uint8_t reserved[6];

    /** 数据引用索引.*/
    uint16_t data_refer_index;

    /** 详细*/
    union
    {
        struct
        {
            /** 预留.*/
            uint16_t reserved1;

            /** 预留.*/
            uint16_t reserved2;

            /** 预留.*/
            uint32_t reserved3[3];

            /** 宽(像素).*/
            uint16_t width;

            /** 高(像素).*/
            uint16_t height;

            /** 
             * 水平分辨率(以整数形式存储的定点数)(DPI).
             * 
             * 高16位: 整数.
             * 低16位: 小数. 
            */
            uint32_t horiz;

            /** 
             * 垂直分辨率(以整数形式存储的定点数)(DPI).
             * 
             * 高16位: 整数.
             * 低16位: 小数.
            */
            uint32_t vert;

            /** 预留.*/
            uint32_t reserved4;

            /** 每个采样中包括的帧数量.*/
            uint16_t frame_count;

            /** 
             * 编码器名字(Pascal string)(32字节).
             * 
             * @note 第一个字符是长度.
            */
            char encname[33];

            /** 位深(bit).*/
            uint16_t depth;

            /** 预留.*/
            uint16_t reserved5;
        } video;

        struct
        {
            /** 版本.*/
            uint16_t version;

            /** */
            uint16_t revision;

            /** 预留.*/
            uint32_t reserved1;

            /** 声道数量.*/
            uint16_t channels; 

            /** 采样大小.*/
            uint16_t sample_size; 

            /** 压缩ID.*/
            uint16_t compression_id;

            /**  包大小.*/
            uint16_t packet_size;   

            /** 采样速率(以整数形式存储的定点数).
             * 
             * 高16位: 整数.
             * 低16位: 小数. 
            */
            uint32_t sample_rate;

            /** V1 描述信息.*/
            struct
            {
                /** 每个包含有几个采样.*/
                uint32_t samples_per_packet;

                /** 每个包字节数.*/
                uint32_t bytes_per_Packet;

                /** 每个帧字节数.*/
                uint32_t bytes_per_frame;

                /** 每个采样字节数.*/
                uint32_t bytes_per_sample;
            } v1;

            /** V2 描述信息.*/
            struct
            {
                /** V2结构大小(包括所有字段).*/
                uint32_t struct_size;

                /** 速率(64bits)(浮点).*/
                double sample_rate;

                /** 声道数量.*/
                uint32_t channels;

                /** 声道数量.*/
                uint32_t reserved;

                /** 每个声道多少bit.*/
                uint32_t bits_per_channel;

                /** 标志*/
                uint32_t format_specific_flags;

                /** 每个包多少字节.*/
                uint32_t bytes_per_audio_packet;

                /** */
                uint32_t lpcm_frames_per_audio_packet;

                /** 扩展信息.*/
                abcdk_object_t *extension;
            } v2;

        } sound;

        struct
        {
            /** 扩展信息.*/
            abcdk_object_t *extension;
        } subtitle;

    } detail;

}abcdk_mp4_atom_sample_desc_t;

/** MP4 avcc(h264)atom.*/
typedef struct _abcdk_mp4_atom_avcc
{
    /** 扩展数据(Global Header). */
    abcdk_object_t *extradata;

} abcdk_mp4_atom_avcc_t;

/** MP4 hvcc(h265)atom.*/
typedef struct _abcdk_mp4_atom_hvcc
{
    /** 扩展数据(Global Header). */
    abcdk_object_t *extradata;

} abcdk_mp4_atom_hvcc_t;

/** MP4 stts atom table.*/
typedef struct _abcdk_mp4_atom_stts_table
{
    /** sample数量.*/
    uint32_t sample_count;

    /** sample时长差(帧与帧之间).*/
    uint32_t sample_duration;

}abcdk_mp4_atom_stts_table_t;

/** MP4 stts(DTS) atom.*/
typedef struct _abcdk_mp4_atom_stts
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 采样数量*/
    uint32_t numbers;

    /** abcdk_mp4_atom_stts_table_t[numbers]*/
    abcdk_mp4_atom_stts_table_t *tables;

}abcdk_mp4_atom_stts_t;

/** MP4 ctts atom table.*/
typedef struct _abcdk_mp4_atom_ctts_table
{
    /** sample数量.*/
    uint32_t sample_count;

    /** PTS相对于DTS的偏移量(帧与帧之间).*/
    int32_t composition_offset;

}abcdk_mp4_atom_ctts_table_t;

/** 
 * MP4 ctts(CTS) atom.
 * 
 * PTS = DTS + CTS
 * 
 * @note 无, 表示没有B帧.
*/
typedef struct _abcdk_mp4_atom_ctts
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 采样数量*/
    uint32_t numbers;

    /** abcdk_mp4_atom_ctts_table_t[numbers]*/
    abcdk_mp4_atom_ctts_table_t *tables;

}abcdk_mp4_atom_ctts_t;

/** MP4 stsc atom table.*/
typedef struct _abcdk_mp4_atom_stsc_table
{
    /** 一组chunk中第一个chunk编号.*/
    uint32_t first_chunk;

    /** 每个chunk内包含的sample数量.*/
    uint32_t samples_perchunk;

    /** sample ID.*/
    uint32_t sample_desc_id;

}abcdk_mp4_atom_stsc_table_t;

/** MP4 stsc atom.*/
typedef struct _abcdk_mp4_atom_stsc
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 采样数量*/
    uint32_t numbers;

    /** abcdk_mp4_atom_stsc_table_t[numbers]*/
    abcdk_mp4_atom_stsc_table_t *tables;

}abcdk_mp4_atom_stsc_t;

/** MP4 stsz atom table.*/
typedef struct _abcdk_mp4_atom_stsz_table
{
    /** sample大小(字节).*/
    uint32_t size;

}abcdk_mp4_atom_stsz_table_t;

/** MP4 stsz atom.*/
typedef struct _abcdk_mp4_atom_stsz
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 
     * 采样大小.
     * 
     * > 0 有效.
     * 
     * = 0 见采样表.
    */
    uint32_t sample_size;

    /** 采样数量*/
    uint32_t numbers;

    /** abcdk_mp4_atom_stsz_table_t[numbers]*/
    abcdk_mp4_atom_stsz_table_t *tables;

}abcdk_mp4_atom_stsz_t;

/** MP4 stco or co64 atom table.*/
typedef struct _abcdk_mp4_atom_stco_table
{
    /** chunk偏移量(以0为基值).*/
    uint64_t offset;

}abcdk_mp4_atom_stco_table_t;

/** MP4 stco or co64 atom.*/
typedef struct _abcdk_mp4_atom_stco
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 采样数量*/
    uint32_t numbers;

    /** abcdk_mp4_atom_stco_table_t[numbers]*/
    abcdk_mp4_atom_stco_table_t *tables;

}abcdk_mp4_atom_stco_t;

/** MP4 stss atom table.*/
typedef struct _abcdk_mp4_atom_stss_table
{
    /** 关健帧编号, 以1为基值).*/
    uint32_t sync;

}abcdk_mp4_atom_stss_table_t;

/** 
 * MP4 stss atom.
 * 
 * @note 无, 全部是关键帧.
*/
typedef struct _abcdk_mp4_atom_stss
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 采样数量*/
    uint32_t numbers;

    /** abcdk_mp4_atom_stss_table_t[numbers]*/
    abcdk_mp4_atom_stss_table_t *tables;

}abcdk_mp4_atom_stss_t;

/** MP4 smhd atom.*/
typedef struct _abcdk_mp4_atom_smhd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 
     * 均衡(以整数形式存储的定点数).
     * 
     * 高8位: 整数.
     * 低8位: 小数. 
    */
    uint16_t balance;

    /** 预留.*/
    uint16_t reserved;
    
}abcdk_mp4_atom_smhd_t;

/** MP4 elst atom table.*/
typedef struct _abcdk_mp4_atom_elst_table
{
    /** */
    uint64_t track_duration;

    /** */
    uint64_t media_time;

    /** 
     * 速率(以整数形式存储的定点数).
     * 
     * 高16位: 整数.
     * 低16位: 小数. 
    */
    uint32_t media_rate;

}abcdk_mp4_atom_elst_table_t;

/** MP4 elst atom.*/
typedef struct _abcdk_mp4_atom_elst
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 采样数量*/
    uint32_t numbers;

    /** abcdk_mp4_atom_elst_table_t[numbers]*/
    abcdk_mp4_atom_elst_table_t *tables;

}abcdk_mp4_atom_elst_t;

#define ABCDK_MP4_ESDS_ES           0x03
#define ABCDK_MP4_ESDS_DEC_CONF     0x04
#define ABCDK_MP4_ESDS_DEC_SP_INFO  0x05
#define ABCDK_MP4_ESDS_SL_CONF      0x06

/** MP4 esds atom.*/
typedef struct _abcdk_mp4_atom_esds
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** */
    uint8_t tag;

    /** tag: 0x03*/
    struct
    {
        /** */
        uint16_t id;

        /*
         * 0x80: depends
         * 0x40: url
         * 0x20: ocr
        */
        uint8_t flags;
        
        /** */
        uint16_t depends;

        /** 
        * URL(Pascal string).
        * 
        * @note 第一个字符是长度.
        */
        char url[260];

        /** */
        uint16_t ocr;

    } es;

    /** tag: 0x04*/
    struct
    {
        /** */
        uint8_t type_id;

        /** 
         * 7~2: Stream Type;
         * 1~0: Up Stream; 
        */
        uint8_t stream_type;

        /** */
        uint32_t buffer_size;

        /** */
        uint32_t max_bitrate;

        /** */
        uint32_t avg_bitrate;
    } dec_conf;

    /** tag: 0x05*/
    struct
    {
        /** 
         * 扩展数据(Global Header).
         * 
         * @note ADTS 在这里.
         */
        abcdk_object_t *extradata;

    } dec_sp_info;

    /** tag: 0x06*/
    struct
    {
        /** */
        uint8_t reserved;
        
    } dec_ld_conf;

} abcdk_mp4_atom_esds_t;

/** MP4 mehd atom.*/
typedef struct _abcdk_mp4_atom_mehd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 时长(秒×时间的刻度值).*/
    uint64_t duration;

}abcdk_mp4_atom_mehd_t;

/** MP4 trex atom.*/
typedef struct _abcdk_mp4_atom_trex
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;
    
    /** TRACK ID.*/
    uint32_t trackid; 

    /** 默认的采样描述索引.*/
    uint32_t sample_desc_idx;

    /** 默认的sample时长.*/
    uint64_t sample_duration;

    /** 默认的sample大小(字节).*/
    uint32_t sample_size; 

    /** 默认的sample标志.*/
    uint32_t sample_flags; 

}abcdk_mp4_atom_trex_t;

/** MP4 mfhd atom.*/
typedef struct _abcdk_mp4_atom_mfhd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** 顺序编号.*/
    uint64_t sequence_number;

}abcdk_mp4_atom_mfhd_t;


/*
 * MP4 tfhd flags
 *
 * @note 从Bento4复制来的.
*/

#define ABCDK_MP4_TFHD_FLAG_BASE_DATA_OFFSET_PRESENT            0x00001
#define ABCDK_MP4_TFHD_FLAG_SAMPLE_DESCRIPTION_INDEX_PRESENT    0x00002
#define ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_DURATION_PRESENT     0x00008
#define ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_SIZE_PRESENT         0x00010
#define ABCDK_MP4_TFHD_FLAG_DEFAULT_SAMPLE_FLAGS_PRESENT        0x00020
#define ABCDK_MP4_TFHD_FLAG_DURATION_IS_EMPTY                   0x10000
#define ABCDK_MP4_TFHD_FLAG_DEFAULT_BASE_IS_MOOF                0x20000 //此标志非常有用, 用于计算数据的偏移量.

/** MP4 tfhd atom.*/
typedef struct _abcdk_mp4_atom_tfhd
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;
    
    /** TRACK ID.*/
    uint32_t trackid; 

    /** 
     * sample偏移量的基值.
     * 
     * @note 如果未设置, sample偏移量以moof起始为基值; 否则, 以此为基值.
    */
    uint64_t base_data_offset;

    /** 默认的sample索引.*/
    uint32_t sample_desc_idx;

    /** 默认的sample时长(秒×时间的刻度值).*/
    uint64_t sample_duration; 

    /** 默认的sample大小.*/
    uint32_t sample_size; 

    /** 默认的标志.*/
    uint32_t sample_flags; 

}abcdk_mp4_atom_tfhd_t;


/** MP4 tfdt atom.*/
typedef struct _abcdk_mp4_atom_tfdt
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** DTS时间基值.*/
    uint64_t base_decode_time;

}abcdk_mp4_atom_tfdt_t;

/*
 * MP4 trun flags
 *
 * @note 从Bento4复制来的.
*/

#define ABCDK_MP4_TRUN_FLAG_DATA_OFFSET_PRESENT                     0x0001
#define ABCDK_MP4_TRUN_FLAG_FIRST_SAMPLE_FLAGS_PRESENT              0x0004
#define ABCDK_MP4_TRUN_FLAG_SAMPLE_DURATION_PRESENT                 0x0100
#define ABCDK_MP4_TRUN_FLAG_SAMPLE_SIZE_PRESENT                     0x0200
#define ABCDK_MP4_TRUN_FLAG_SAMPLE_FLAGS_PRESENT                    0x0400
#define ABCDK_MP4_TRUN_FLAG_SAMPLE_COMPOSITION_TIME_OFFSET_PRESENT  0x0800

#define ABCDK_MP4_TRUN_FLAG_OPTION_RESERVED              \
    (~(ABCDK_MP4_TRUN_FLAG_DATA_OFFSET_PRESENT |         \
       ABCDK_MP4_TRUN_FLAG_FIRST_SAMPLE_FLAGS_PRESENT) & \
     0x0000FF)

#define ABCDK_MP4_TRUN_FLAG_SAPLME_RESERVED                          \
    (~(ABCDK_MP4_TRUN_FLAG_SAMPLE_DURATION_PRESENT |                 \
       ABCDK_MP4_TRUN_FLAG_SAMPLE_SIZE_PRESENT |                     \
       ABCDK_MP4_TRUN_FLAG_SAMPLE_FLAGS_PRESENT |                    \
       ABCDK_MP4_TRUN_FLAG_SAMPLE_COMPOSITION_TIME_OFFSET_PRESENT) & \
     0x00FF00)

/** MP4 trun atom table.*/
typedef struct _abcdk_mp4_atom_trun_table
{
    /** 
     * sample时长.
     *  
     * @note 未设置时, 使用默认值(在tfhd或trex中).
    */
    uint32_t sample_duration;

    /** 
     * sample大小(字节).
     * 
     * @note 未设置时, 使用默认值(在tfhd或trex中).
    */
    uint32_t sample_size;

    /** 
     * sample标志.
     * 
     * @note 未设置时, 使用默认值(在tfhd或trex中).
    */
    uint32_t sample_flags;

    /** 
     * PTS相对于DTS的偏移量(帧与帧之间).
     * 
     * @note 未设置时, 使用默认值(0).
    */
    int32_t composition_offset;

}abcdk_mp4_atom_trun_table_t;

/** MP4 trun atom.*/
typedef struct _abcdk_mp4_atom_trun
{
    /** 版本.*/
    uint8_t version;

    /** 标志.
     *      
     * 0～7位: 1 选项字段‘有’, 0 选项字段‘无’.
     * 8～15位: 1 采样表字段‘有’, 0 采样表字段‘无’.
    */
    uint32_t flags;

    /** 采样数量*/
    uint32_t numbers;

    /** 
     * sample偏移量(基值在tfhd中).
     * 
     * @note 未设置时, 使用默认值(0).
    */
    uint32_t data_offset;

    /** 
     * 第一个sample标志.
     * 
     * @note 未设置时, 使用默认值(0).
    */
    uint32_t first_sample_flags;

    /** abcdk_mp4_atom_trun_table_t[numbers]*/
    abcdk_mp4_atom_trun_table_t *tables;

}abcdk_mp4_atom_trun_t;

/** MP4 mfro atom.*/
typedef struct _abcdk_mp4_atom_mfro
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** */
    uint64_t size;

}abcdk_mp4_atom_mfro_t;

/** MP4 tfra atom table.*/
typedef struct _abcdk_mp4_atom_tfra_table
{
    /** */
    uint64_t time;

    /** */
    uint64_t moof_offset;

    /** */
    uint32_t traf_number;

    /** */
    uint32_t trun_number;

    /** */
    uint32_t sample_number;

}abcdk_mp4_atom_tfra_table_t;

/** MP4 tfra atom.*/
typedef struct _abcdk_mp4_atom_tfra
{
    /** 版本.*/
    uint8_t version;

    /** 标志.*/
    uint32_t flags;

    /** TRACK ID.*/
    uint32_t trackid;

    /** 
     * traf NUMBER size.
     * 
     * 0 is uint8.
     * 1 is uint16.
     * 2 is uint24.
     * 3 is uint32.
    */
    uint8_t length_size_traf_num;

    /** 
     * trun NUMBER size.
     * 
     * 0 is uint8.
     * 1 is uint16.
     * 2 is uint24.
     * 3 is uint32.
    */
    uint8_t length_size_trun_num;

    /** 
     * sample NUMBER size.
     * 
     * 0 is uint8.
     * 1 is uint16.
     * 2 is uint24.
     * 3 is uint32.
    */
    uint8_t length_size_sample_num;

    /** 采样数量*/
    uint32_t numbers;

    /** abcdk_mp4_atom_tfra_table_t[numbers]*/
    abcdk_mp4_atom_tfra_table_t *tables;

}abcdk_mp4_atom_tfra_t;

/** MP4 unknown atom.*/
typedef struct _abcdk_mp4_atom_unknown
{
    /** 原始数据. */
    abcdk_object_t *rawbytes;

} abcdk_mp4_atom_unknown_t;


/** MP4 atom.*/
typedef struct _abcdk_mp4_atom
{
    /** 长度(字节, 头部+数据).*/
    uint64_t size;

    /** 类型.*/
    abcdk_mp4_tag_t type;

    /** 头部偏移量(字节, 以0为基值).*/
    uint64_t off_head;

    /** 数据偏移量(字节, 以0为基值).*/
    uint64_t off_data;

    /** 
     * 是否有数据.
     * 
     * 0 无(容器), !0 有.
    */
    int have_data;

    /**
     * 数据.
     * 
     * @note 详细结构见各类型定义.
    */
    union 
    {
        abcdk_mp4_atom_ftyp_t ftyp;
        abcdk_mp4_atom_mvhd_t mvhd;
        abcdk_mp4_atom_tkhd_t tkhd;
        abcdk_mp4_atom_mdhd_t mdhd;
        abcdk_mp4_atom_hdlr_t hdlr;
        abcdk_mp4_atom_vmhd_t vmhd;
        abcdk_mp4_atom_dref_t dref;
        abcdk_mp4_atom_stsd_t stsd;
        abcdk_mp4_atom_sample_desc_t sample_desc;
        abcdk_mp4_atom_avcc_t avcc;
        abcdk_mp4_atom_hvcc_t hvcc;
        abcdk_mp4_atom_esds_t esds;
        abcdk_mp4_atom_stts_t stts;
        abcdk_mp4_atom_ctts_t ctts;
        abcdk_mp4_atom_stsc_t stsc;
        abcdk_mp4_atom_stsz_t stsz;
        abcdk_mp4_atom_stco_t stco;
        abcdk_mp4_atom_stss_t stss;
        abcdk_mp4_atom_smhd_t smhd;
        abcdk_mp4_atom_elst_t elst;
        abcdk_mp4_atom_mehd_t mehd;
        abcdk_mp4_atom_trex_t trex;
        abcdk_mp4_atom_mfhd_t mfhd;
        abcdk_mp4_atom_tfhd_t tfhd;
        abcdk_mp4_atom_tfdt_t tfdt;
        abcdk_mp4_atom_trun_t trun;
        abcdk_mp4_atom_mfro_t mfro;
        abcdk_mp4_atom_tfra_t tfra;
        abcdk_mp4_atom_unknown_t unknown;
    }data;
    
} abcdk_mp4_atom_t;

/**
 * 创建atom.
*/
abcdk_tree_t *abcdk_mp4_alloc();

/** 
 * 查找atom.
 * 
 * @param index 索引(以1为基值).
 * @param recursive 0 仅查找根和一级子节点, !0 递归查找所有子节点.
 * 
 * @return !NULL(0) 成功, NULL(0) 失败.
*/
abcdk_tree_t *abcdk_mp4_find(abcdk_tree_t *root,abcdk_mp4_tag_t *type,size_t index,int recursive);

/** 
 * 查找atom.
 * 
 * @param type 类型(大端字节序).
 * 
 * @return !NULL(0) 成功, NULL(0) 失败.
*/
abcdk_tree_t *abcdk_mp4_find2(abcdk_tree_t *root,uint32_t type,size_t index,int recursive);

/**
 * 打印结构.
*/
void abcdk_mp4_dump(FILE *fd, abcdk_tree_t *root);

/**
 * 查询数据包所属的chunk编号(在stco中, 以1为基值), 在chunk内部的偏移量(第几个, 以1为基值), 和所属的ID.
 * 
 * @param sample 数据包编号(在stsz中, 以1为基值).
 * @param chunk chunk编号的指针, 返回前填写.
 * @param offset 偏移量的指针, 返回前填写.
 * @param id ID的指针, 返回前填写.
 * 
 * @return 0 成功, -1 失败(值错误).
*/
int abcdk_mp4_stsc_tell(abcdk_mp4_atom_stsc_t *stsc,uint32_t sample,uint32_t *chunk,uint32_t *offset,uint32_t *id);

/**
 * 查询数据包所在chunk中的偏移量(在stco中, 以0为基值), 和数据包的长度.
 * 
 * @param off_chunk 数据包所在chunk的内部编号(第几个, 以1为基值).
 * @param sample 数据包编号(在stsz中, 以1为基值).
 * @param offset 偏移量的指针, 返回前填写.
 * @param size 长度的指针, 返回前填写.
 * 
 * @return 0 成功, -1 失败(超出范围).
*/
int abcdk_mp4_stsz_tell(abcdk_mp4_atom_stsz_t *stsz, uint32_t off_chunk, uint32_t sample, uint32_t *offset, uint32_t *size);

/**
 * 查询数据包的DTS, 和时长.
 * 
 * @param sample 数据包编号(在stsz中, 以1为基值).
 * @param dts DTS的指针, 返回前填写.
 * @param duration 时长的指针, 返回前填写.
 * 
 * @return 0 成功, -1 失败(超出范围).
*/
int abcdk_mp4_stts_tell(abcdk_mp4_atom_stts_t *stts, uint32_t sample, uint64_t *dts, uint32_t *duration);

/**
 * 查询数据包的CTS(PTS相对于DTS的偏移量).
 * 
 * @param sample 数据包编号(在stsz中, 以1为基值).
 * @param offset CTS的指针, 返回前填写.
 * 
 * @return 0 成功, -1 失败(超出范围).
*/
int abcdk_mp4_ctts_tell(abcdk_mp4_atom_ctts_t *ctts,uint32_t sample,  int32_t *offset);

/**
 * 查询是否为关键帧.
 * 
 * @return 0 成功(是), -1 失败(否, 或超出范围).
*/
int abcdk_mp4_stss_tell(abcdk_mp4_atom_stss_t *stss,uint32_t sample);

__END_DECLS

#endif //ABCDK_MP4_ATOM_H