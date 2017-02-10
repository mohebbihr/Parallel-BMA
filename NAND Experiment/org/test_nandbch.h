/*
*  This file contains the routines for testing the NAND BCH encoder/decoder.
*
*
*/

#ifndef _TEST_NANDBCH_H
#define _TEST_NANDBCH_H

#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include "bch.h"
#include "nand_bch.h"
#include "poisson.h"

#define ENOMEM          12      /* Out of memory */
#define BITS_PER_BYTE           8
#define ERR_NO_NUM -1
#define ERR_NO_MEM -2

static struct nand_ecclayout nand_oob_8 = {
	.eccbytes = 3,
	.eccpos = {0, 1, 2},
	.oobfree = {
		{.offset = 3,
		 .length = 2},
		{.offset = 6,
		 .length = 2} }
};

static struct nand_ecclayout nand_oob_16 = {
	.eccbytes = 6,
	.eccpos = {0, 1, 2, 3, 6, 7},
	.oobfree = {
		{.offset = 8,
		 .length = 8} }
};

static struct nand_ecclayout nand_oob_64 = {
	.eccbytes = 24,
	.eccpos = {
		   40, 41, 42, 43, 44, 45, 46, 47,
		   48, 49, 50, 51, 52, 53, 54, 55,
		   56, 57, 58, 59, 60, 61, 62, 63},
	.oobfree = {
		{.offset = 2,
		 .length = 38} }
};

static struct nand_ecclayout nand_oob_128 = {
	.eccbytes = 47, //48
	.eccpos = {
		   80, 81, 82, 83, 84, 85, 86, 87,
		   88, 89, 90, 91, 92, 93, 94, 95,
		   96, 97, 98, 99, 100, 101, 102, 103,
		   104, 105, 106, 107, 108, 109, 110, 111,
		   112, 113, 114, 115, 116, 117, 118, 119,
		   120, 121, 122, 123, 124, 125, 126, 127},
	.oobfree = {
		{.offset = 2,
		 .length = 78} }
};

void print_array(void *data, unsigned int size){
		
	unsigned int i;
	char * p = (char *) data;
	
	for(i=0; i<size; i++){		
		printf("data[%d] : %d \n", i , p[i]);
	}				
	
}

/**
 * change_bit - Toggle a bit in memory
 * @nr: the bit to change
 * @addr: the address to start counting from
 *
 */
static inline void change_bit(unsigned int nr, void *addr)
{
	unsigned int byteindex = nr / (sizeof(char) * BITS_PER_BYTE);
	unsigned int bitindex = nr % (sizeof(char) * BITS_PER_BYTE);
	
	char * p = (char *) addr;
	p += byteindex;
	*p ^= (1<<bitindex);
}

/**
* bit_error: flip multiple bits in the data. The function use poissonRandom to insert errors randomly on data.
* @error_data : output - a pointer to the data with error
* @correct_data : input pointer to the buffer
* @size: the size of input in bytes
* @numerr : the number of errors that we want to insert. 
*/
void bit_error(void *error_data, void *correct_data, size_t size, size_t numerr)
{
	unsigned int k=1;
	unsigned int offset[numerr];	
	
	offset[0] = Random() % (size * BITS_PER_BYTE);		
	
	while(k < numerr){		
		offset[k] = Random() % (size * BITS_PER_BYTE);		
		if(offset[k-1] != offset[k]) k++;
	}

	memcpy(error_data, correct_data, size);

	for(k=0; k<numerr; k++){		
		change_bit(offset[k], error_data);
	}
		
}

void bit_error_poisson(void *error_data, void *correct_data, size_t size, size_t numerr, int lambda)
{
	unsigned int k=1;
	unsigned int offset[numerr];	

	offset[0] = poissonRandom(lambda) % (size * BITS_PER_BYTE);		
	
	while(k < numerr){
		offset[k] = poissonRandom(lambda) % (size * BITS_PER_BYTE);				
		if(offset[k-1] != offset[k]) k++;
	}

	memcpy(error_data, correct_data, size);

	for(k=0; k<numerr; k++){
		change_bit(offset[k], error_data);
	}
		
}

/**
* bit_error_detect: This function correct the errors in the data using nand_bch_calculate_ecc and nand_bch_correct_data functions.
* @mtd: the pointer to the mtd_info control structure for a nand device
* @error_data : a pointer to the data with error
* @error_ecc: a pointer to the ecc of the error_data
* @ecc_size: the size of ecc in bytes
*/
int bit_error_detect(struct mtd_info *mtd, void *error_data, void *error_ecc, const size_t ecc_size)
{	
	unsigned char calc_ecc[ecc_size];
	int ret;
	// we calculate the ecc from the error_data and we compare it with the ecc that we read from the media (error_ecc)
	nand_bch_calculate_ecc(mtd,error_data,calc_ecc);
	ret = nand_bch_correct_data(mtd, error_data, error_ecc, calc_ecc);    

	return (ret == -1) ? 0 : -EINVAL;
}

int bit_error_detect_SSE(struct mtd_info *mtd, void *error_data, void *error_ecc, const size_t ecc_size)
{	
	unsigned char calc_ecc[ecc_size];
	int ret;
	// we calculate the ecc from the error_data and we compare it with the ecc that we read from the media (error_ecc)
	nand_bch_calculate_ecc(mtd,error_data,calc_ecc);
	ret = nand_bch_correct_data_SSE(mtd, error_data, error_ecc, calc_ecc);    	

	return (ret == -1) ? 0 : -EINVAL;
}

/**
* random_bytes: fill the data with random data
* @data: the buffers
* @size: the size of buffer in bytes
*/
void random_bytes(void *data, unsigned int size){
		
	struct drand48_data buffer; // used as a seed for generating random number
	long int random_value; 
	struct timeval tv; 
	unsigned int i;
	char * p = (char *) data;
	
	for(i=0; i<size; i++){
		gettimeofday(&tv, NULL);
		srand48_r(tv.tv_sec + tv.tv_usec, &buffer);
		lrand48_r(&buffer, &random_value);
		*p = (char)(random_value % UCHAR_MAX);		
		p++;
	}				
	
}

/*int distinct_random (int size) {
    int i, n;
    static int numNums = 0;
    static int *numArr = NULL;

    if(size < 0) return -1;

    if (numArr == NULL){
        numArr = malloc (sizeof(int) * size);
        for (i = 0; i  < size; i++)
    		numArr[i] = i;
	numNums = size;
    }

    if (numNums == 0)
       return ERR_NO_NUM;

    n = rand() % numNums;
    i = numArr[n];
    numArr[n] = numArr[numNums-1];
    numNums--;
    if (numNums == 0) {
        free (numArr);
        numArr = NULL;
    }

    return i;
}*/

void fill_distinct_random(int * arr, int size, int randsize, int lambda){
	int i;
	for(i=0; i< size; i++)
		arr[i] = distinct_random(randsize, lambda);
		//arr[i] = Random() % randsize;
}

void nand_ecc_ctrl_free(struct nand_ecc_ctrl *ecc){	
	nand_bch_free(ecc->priv);		
}

struct nand_ecc_ctrl nand_ecc_ctrl_init(struct mtd_info *mtd, unsigned int size, unsigned int bytes, struct nand_ecclayout *layout){
	struct nand_ecc_ctrl ecc;
	ecc.size = size;
	ecc.bytes = bytes;
	ecc.layout = layout;
	ecc.priv = nand_bch_init(mtd,size,bytes,&layout);
	if(!ecc.priv)
		goto fail;
	return ecc;
fail:
	nand_ecc_ctrl_free(&ecc);
	return ecc;
}

void nand_chip_free(struct nand_chip * chip){
	if (chip) {
		nand_ecc_ctrl_free(&chip->ecc);				
	}
}

struct nand_chip * nand_chip_init(struct mtd_info *mtd, unsigned int size, unsigned int bytes, struct nand_ecclayout *layout){
	struct nand_chip *chip = NULL;
	chip = malloc(sizeof(struct nand_chip));
	if(!chip)
		goto fail;
	chip->ecc = nand_ecc_ctrl_init(mtd,size,bytes,layout);	
	if(!chip->ecc.priv)
		goto fail;
	return chip;
fail:
	nand_chip_free(chip);
	return NULL;
}

/**
 * mtd_info_free - [mtd_info Interface] Release mtd_info resources
 * @nbc:	NAND BCH control structure
 */
void mtd_info_free(struct mtd_info *mtd)
{
	if (mtd) {		
		nand_chip_free(mtd->priv);
		free(mtd);		
	}
}

/**
* mtd_info_init: Initialize a mtd_info structure
* @writesize: 
*/
struct mtd_info * mtd_info_init(unsigned int writesize, unsigned int oobsize, unsigned int eccsize, unsigned int eccbytes){
	
	struct mtd_info *mtd = NULL;
	mtd = malloc(sizeof(struct mtd_info));
	if(!mtd)
		goto fail;
	
	mtd->writesize = writesize;	
	mtd->oobsize = oobsize;	
	switch(mtd->oobsize){
		case 8:
			mtd->priv = nand_chip_init(mtd,eccsize,eccbytes,&nand_oob_8);
			break;
		case 16:
			mtd->priv = nand_chip_init(mtd,eccsize,eccbytes,&nand_oob_16);
			break;
		case 64:
			mtd->priv = nand_chip_init(mtd,eccsize,eccbytes,&nand_oob_64);
			break;
		case 128:
			mtd->priv = nand_chip_init(mtd,eccsize,eccbytes,&nand_oob_128);
			break;
		default:
			printf("Error in ecclayout, it must be 8,16,64 or 128 \n");
			goto fail;
	}
	if (!mtd->priv){
		printf("BCH ECC initialization failed!\n");
		goto fail;
	}
	
	return mtd;	
	
fail:
	mtd_info_free(mtd);
	return NULL;
}

void cmp_array(void *data1, void * data2, unsigned int size){
		
	unsigned int i;
	char * p1 = (char *) data1;
	char * p2 = (char *) data2;
	
	for(i=0; i<size; i++){		
		if(p1[i] != p2[i])
			printf("two array's are not equal, %d - %d\n", p1[i], p2[i]);
	}				
	
}

/**
* nandbch_test: The main function for doing the test. 
* @bufsize: The buffer size of data in bytes
* @eccsize: ecc block size in bytes
* @oobsize:	the layout number (8,16,64,128)
* @eccbytes:	ecc length in bytes
* @numerr: the number of errors created in each block for test
*/	
int nandbch_test(struct mtd_info *mtd, void *correct_data, void *correct_ecc, void *error_data, void *error_ecc, unsigned int bufsize, unsigned int numerr)
{	
	int err = 0;

	nand_bch_calculate_ecc(mtd,correct_data,correct_ecc);
	bit_error(error_data,correct_data,bufsize,numerr);
	nand_bch_calculate_ecc(mtd,error_data,error_ecc);
	err = nand_bch_correct_data(mtd,error_data,error_ecc,correct_ecc);	
	
	return err;
}


void nandbch_test_prepare(struct mtd_info *mtd, void *correct_data, void *correct_ecc, void *error_data, void *error_ecc, unsigned int bufsize, unsigned int numerr, int haserror)
{
        nand_bch_calculate_ecc(mtd,correct_data,correct_ecc);
	if(haserror == 1)
        	bit_error(error_data,correct_data,bufsize,numerr);
	else
		bit_error(error_data,correct_data,bufsize,0);
        nand_bch_calculate_ecc(mtd,error_data,error_ecc);
}

#endif
