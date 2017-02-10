/* nand-exp.c: This file contains the code for encoding/decoding the nand data using bch method.
*
*
*
*
*/

#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include "test_nandbch.h"
#include "bch.h"

int memsize = 1; // in MB scale
int lambda = 1;

void init(int argc, char* argv[]){
  int i;

  if (argc < 3){
    printf("Usage: ./experiment.o -m total memory MB -l lambda\n");    
    exit(1);
  }
  
  if (argc >= 3){
    i = 1;
    while (i<argc){
      if (!strcmp(argv[i],"-m")) sscanf(argv[i+1],"%d",&memsize);
	  if (!strcmp(argv[i],"-l")) sscanf(argv[i+1],"%d",&lambda);
      i += 2;
    }
  }     
  
}

void main(int argc, char *argv[]){						
		
	//Error Recovery fails when, bufsize are different than writesize and eccsize.
	// In this experiment writesize = eccsize
	struct timeval tv1, tv2;
	int sec, usec;
	int iterations = 1024; // 1
	int bufsize = 2048; // 32
	int eccsize = 2048; //32
	int eccbytes = 47; // 3
	int oobsize = 128; //8
	int m,t,i, j=0;
	int pageindex = 0;
	int numpage =0, numerrpages =0;
	int err; 
	struct mtd_info *mtd;	
	int * errindexes;
	void *error_data;
	void *error_ecc;
	void *correct_data;
	void *correct_ecc;

	init(argc,argv);   
	
	// initialize the mtd_info		
	mtd = mtd_info_init(eccsize,oobsize,eccsize,eccbytes);
		
	if(!mtd){
		printf("Can not start the test\n");
		return;
	}
	
	m = fls(1+8*eccsize);
	t = (eccbytes*8)/m;	
	numpage = iterations * memsize;
	
	numerrpages = (int)ceil(numpage * 0.1);
	errindexes = (int *)malloc(numerrpages * sizeof(int));
	fill_distinct_random(errindexes, numerrpages, numpage, lambda);
	BubbleSort(errindexes, numerrpages);
	// allocate memory for buffers
	error_data = malloc(bufsize * numpage);
	error_ecc = malloc(eccsize * numpage);
	correct_data = malloc(bufsize * numpage);
	correct_ecc = malloc(eccsize * numpage);
	
	printf("SSE Code \nExperiment starts with - m: %d, t: %d, numpage: %d\n",m,t,numpage);
	// fill the data with random numbers
	random_bytes(correct_data,bufsize * numpage);	
	
	for(i=0; i<numpage && j< numerrpages; i++){

		if(i== errindexes[j]){
			//printf("error created at index: %d \n", i);
			nandbch_test_prepare(mtd,(void *)&correct_data[i * bufsize],(void *)&correct_ecc[i * eccsize],(void *)&error_data[i * bufsize],(void *)&error_ecc[i * eccsize],bufsize,t,1);
			j++;
		}else
			nandbch_test_prepare(mtd,(void *)&correct_data[i * bufsize],(void *)&correct_ecc[i * eccsize],(void *)&error_data[i * bufsize],(void *)&error_ecc[i * eccsize],bufsize,t,0);
	
	}
	
	gettimeofday(&tv1,NULL);	
	for(i=0; i<numpage; i++){
		err = nand_bch_correct_data_SSE(mtd,(void *)&error_data[i * bufsize],(void *)&error_ecc[i * eccsize],(void *)&correct_ecc[i * eccsize]);
		//printf("i: %d, err: %d \n", i, err);
	}
	gettimeofday(&tv2,NULL);
	sec = (int) (tv2.tv_sec-tv1.tv_sec);
	usec = (int) (tv2.tv_usec-tv1.tv_usec);
	if (usec < 0){
			sec--;
			usec += 1000000;
	}
	printf("SSE Total time for %d MB : %f sec\n",memsize * 2,sec+usec/1000000.0);

	//free the momory
	mtd_info_free(mtd);	
	free(error_data);
	free(error_ecc);
	free(correct_data);
	free(correct_ecc);
			
}
