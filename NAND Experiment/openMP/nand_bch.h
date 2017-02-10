/*
 * nand-bch.h: contains the files for NAND decoder/encoder using BCH algorithm. 
 *
 * 
 *
 * 
 */

#ifndef _NAND_BCH_H
#define _NAND_BCH_H

#include "bch.h"

#define MTD_MAX_OOBFREE_ENTRIES_LARGE	32
#define MTD_MAX_ECCPOS_ENTRIES_LARGE	640

typedef unsigned int __u32;

struct nand_oobfree {
    __u32 offset;
    __u32 length;
};

/*
 * Internal ECC layout control structure. For historical reasons, there is a
 * similar, smaller struct nand_ecclayout_user (in mtd-abi.h) that is retained
 * for export to user-space via the ECCGETLAYOUT ioctl.
 * nand_ecclayout should be expandable in the future simply by the above macros.
 */
struct nand_ecclayout {
	__u32 eccbytes;
	__u32 eccpos[MTD_MAX_ECCPOS_ENTRIES_LARGE];
	__u32 oobavail;
	struct nand_oobfree oobfree[MTD_MAX_OOBFREE_ENTRIES_LARGE];
};

struct mtd_info {
	/* Minimal writable flash unit size. In case of NOR flash it is 1 (even
	 * though individual bits can be cleared), in case of NAND flash it is
	 * one NAND page (or half, or one-fourths of it), in case of ECC-ed NOR
	 * it is of ECC block size, etc. It is illegal to have writesize = 0.
	 * Any driver registering a struct mtd_info must ensure a writesize of
	 * 1 or larger.
	 */
	unsigned int writesize;
	unsigned int oobsize;   // Amount of OOB data per block (e.g. 16)	
	void *priv;

};

/**
 * struct nand_ecc_ctrl - Control structure for ECC
 * @size:	data bytes per ECC step
 * @bytes:	ECC bytes per step
 * @priv:	pointer to private ECC control data
 */
struct nand_ecc_ctrl {	
	unsigned int size;
	unsigned int bytes;	
	void *priv;
	struct nand_ecclayout   *layout;
};

/**
 * struct nand_chip - NAND Private Flash Chip Data
 * @ecc:		[BOARDSPECIFIC] ECC control structure
 * 
 */
struct nand_chip {
	struct nand_ecc_ctrl ecc;	
};

/**
 * struct nand_bch_control - private NAND BCH control structure 
 * @bch:       BCH control structure
 * @ecclayout: private ecc layout for this BCH configuration
 * @errloc:    error location array
 * @eccmask:   XOR ecc mask, allows erased pages to be decoded as valid
 */
struct nand_bch_control {
	struct bch_control   *bch;
	struct nand_ecclayout ecclayout;
	unsigned int         *errloc;
	unsigned char        *eccmask;
};

inline int mtd_nand_has_bch(void) { return 1; }

/**
 * nand_bch_calculate_ecc - [NAND Interface] Calculate ECC for data block
 * @mtd:	MTD block structure
 * @buf:	input buffer with raw data
 * @code:	output buffer with ECC, it contains just ECC.
 */
int nand_bch_calculate_ecc(struct mtd_info *mtd, const unsigned char *buf, unsigned char *code)
{
	const struct nand_chip *chip = mtd->priv;
	struct nand_bch_control *nbc = chip->ecc.priv;
	unsigned int i;

	memset(code, 0, chip->ecc.bytes);
	encode_bch(nbc->bch, buf, chip->ecc.size, code);

	/* apply mask so that an erased page is a valid codeword */
	for (i = 0; i < chip->ecc.bytes; i++)
		code[i] ^= nbc->eccmask[i];

	return 0;
}

/**
 * nand_bch_correct_data - [NAND Interface] Detect and correct bit error(s)
 * @mtd:	MTD block structure
 * @buf:	raw data read from the chip
 * @read_ecc:	ECC from the chip
 * @calc_ecc:	the ECC calculated from raw data
 *
 * Detect and correct bit errors for a data byte block
 */
int nand_bch_correct_data(struct mtd_info *mtd, unsigned char *buf, unsigned char *read_ecc, unsigned char *calc_ecc)
{
	const struct nand_chip *chip = mtd->priv;
	struct nand_bch_control *nbc = chip->ecc.priv;
	unsigned int *errloc = nbc->errloc;
	int i, count;

	count = decode_bch(nbc->bch, NULL, chip->ecc.size, read_ecc, calc_ecc, NULL, errloc);
	
	if (count > 0) {
		for (i = 0; i < count; i++) {
			if (errloc[i] < (chip->ecc.size*8))
				/* error is located in data, correct it */
				buf[errloc[i] >> 3] ^= (1 << (errloc[i] & 7));
			/* else error in ecc, no action needed */
			
		}
	} else if (count < 0) {
		//printf("ecc unrecoverable error\n");
		count = -1;
	}
	return count;
}

int nand_bch_correct_data_SSE(struct mtd_info *mtd, unsigned char *buf, unsigned char *read_ecc, unsigned char *calc_ecc)
{
	const struct nand_chip *chip = mtd->priv;
	struct nand_bch_control *nbc = chip->ecc.priv;
	unsigned int *errloc = nbc->errloc;
	int i, count;

	count = decode_bch_SSE(nbc->bch, NULL, chip->ecc.size, read_ecc, calc_ecc, NULL, errloc);
	
	if (count > 0) {
		for (i = 0; i < count; i++) {
			if (errloc[i] < (chip->ecc.size*8))
				/* error is located in data, correct it */
				buf[errloc[i] >> 3] ^= (1 << (errloc[i] & 7));
			/* else error in ecc, no action needed */
			
		}
	} else if (count < 0) {
		//printf("ecc unrecoverable error\n");
		count = -1;
	}
	return count;
}

void nand_bch_free(struct nand_bch_control *nbc);

/**
 * nand_bch_init - [NAND Interface] Initialize NAND BCH error correction
 * @mtd:	MTD block structure
 * @eccsize:	ecc block size in bytes
 * @eccbytes:	ecc length in bytes
 * @ecclayout:	output default layout
 *
 * Returns:
 *  a pointer to a new NAND BCH control structure, or NULL upon failure
 *
 * Initialize NAND BCH error correction. Parameters @eccsize and @eccbytes
 * are used to compute BCH parameters m (Galois field order) and t (error
 * correction capability). @eccbytes should be equal to the number of bytes
 * required to store m*t bits, where m is such that 2^m-1 > @eccsize*8.
 *
 * Example: to configure 4 bit correction per 512 bytes, you should pass
 * @eccsize = 512  (thus, m=13 is the smallest integer such that 2^m-1 > 512*8)
 * @eccbytes = 7   (7 bytes are required to store m*t = 13*4 = 52 bits)
 */
struct nand_bch_control *
nand_bch_init(struct mtd_info *mtd, unsigned int eccsize, unsigned int eccbytes,
	      struct nand_ecclayout **ecclayout)
{
	unsigned int m, t, eccsteps, i;
	struct nand_ecclayout *layout;
	struct nand_bch_control *nbc = NULL;
	unsigned char *erased_page;

	if (!eccsize || !eccbytes) {
		printf("ecc parameters not supplied\n");
		goto fail;
	}

	m = fls(1+8*eccsize);
	t = (eccbytes*8)/m;

	nbc = malloc(sizeof(*nbc));
	if (!nbc)
		goto fail;

	nbc->bch = init_bch(m, t, 0);
	if (!nbc->bch)
		goto fail;

	/* verify that eccbytes has the expected value */
	if (nbc->bch->ecc_bytes != eccbytes) {
		printf("invalid eccbytes %u, should be %u\n",
		       eccbytes, nbc->bch->ecc_bytes);
		goto fail;
	}

	eccsteps = mtd->writesize/eccsize;

	/* if no ecc placement scheme was provided, build one */
	if (!*ecclayout) {

		/* handle large page devices only */
		if (mtd->oobsize < 64) {
			printf("must provide an oob scheme for "
			       "oobsize %d\n", mtd->oobsize);
			goto fail;
		}

		layout = &nbc->ecclayout;
		layout->eccbytes = eccsteps*eccbytes;

		/* reserve 2 bytes for bad block marker */
		if (layout->eccbytes+2 > mtd->oobsize) {
			printf("no suitable oob scheme available "
			       "for oobsize %d eccbytes %u\n", mtd->oobsize,
			       eccbytes);
			goto fail;
		}
		/* put ecc bytes at oob tail */
		for (i = 0; i < layout->eccbytes; i++)
			layout->eccpos[i] = mtd->oobsize-layout->eccbytes+i;

		layout->oobfree[0].offset = 2;
		layout->oobfree[0].length = mtd->oobsize-2-layout->eccbytes;

		*ecclayout = layout;
	}

	/* sanity checks */
	if (8*(eccsize+eccbytes) >= (1 << m)) {
		printf("eccsize %u is too large\n", eccsize);
		goto fail;
	}
	if ((*ecclayout)->eccbytes != (eccsteps*eccbytes)) {
		printf("invalid ecc layout\n");
		goto fail;
	}

	nbc->eccmask = malloc(eccbytes);
	nbc->errloc = malloc(t*sizeof(*nbc->errloc));
	if (!nbc->eccmask || !nbc->errloc)
		goto fail;
	/*
	 * compute and store the inverted ecc of an erased ecc block
	 */
	erased_page = malloc(eccsize);
	if (!erased_page)
		goto fail;

	memset(erased_page, 0xff, eccsize);
	memset(nbc->eccmask, 0, eccbytes);
	encode_bch(nbc->bch, erased_page, eccsize, nbc->eccmask);
	free(erased_page);

	for (i = 0; i < eccbytes; i++)
		nbc->eccmask[i] ^= 0xff;

	return nbc;
fail:
	nand_bch_free(nbc);
	return NULL;
}


/**
 * nand_bch_free - [NAND Interface] Release NAND BCH ECC resources
 * @nbc:	NAND BCH control structure
 */
void nand_bch_free(struct nand_bch_control *nbc)
{
	if (nbc) {
		free_bch(nbc->bch);
		free(nbc->errloc);
		free(nbc->eccmask);
		free(nbc);
	}
}




#endif /* __MTD_NAND_BCH_H__ */
