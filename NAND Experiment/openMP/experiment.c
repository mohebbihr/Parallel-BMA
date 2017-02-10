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

int NUM_THREADS=256; //more than 128 threads, I've got segmentation fault
int memsize = 1; // in MB scale
int lambda = 1;

void init(int argc, char* argv[]){
  int i;

  if (argc < 3){
    printf("Usage: ./experiment.o -m total memory MB -l lambda -t num_threads\n");    
    exit(1);
  }
  
  if (argc >= 3){
    i = 1;
    while (i<argc){
      if (!strcmp(argv[i],"-m")) sscanf(argv[i+1],"%d",&memsize);
	  if (!strcmp(argv[i],"-l")) sscanf(argv[i+1],"%d",&lambda);
	  if (!strcmp(argv[i],"-t")) sscanf(argv[i+1],"%d",&NUM_THREADS);
      i += 2;
    }
  }     
  
}

void main(int argc, char *argv[]){						
		
	//Error Recovery fails when, bufsize are different than writesize and eccsize.
	// In this experiment writesize = eccsize
	struct timeval tv1, tv2;
	int sec, usec;
	int iterations = 1024; // 1024	
	int bufsize = 2048; // 32
	int eccsize = 2048; //32
	int eccbytes = 47; // 3
	int oobsize = 128; //8
	int m,t,i,j,tid,my_first,my_last,j_first, j_last,nThreads;
	int pageindex = 0;
	int numpage =0, numerrpages =0;
	int err; 
	struct mtd_info *mtd, *mtd2;	
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
	//
	//for(i=0; i<numerrpages; i++)
	//	printf("errindexes[%d]: %d \n", i, errindexes[i]);
	// allocate memory for buffers
	error_data = malloc(bufsize * numpage);
	error_ecc = malloc(eccsize * numpage);
	correct_data = malloc(bufsize * numpage);
	correct_ecc = malloc(eccsize * numpage);
		
	omp_set_num_threads(NUM_THREADS);	 
	printf("OpenMP \nExperiment starts with - m: %d, t: %d, numpage: %d, num_threads: %d\n",m,t,numpage,NUM_THREADS);
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

	/*#pragma omp parallel private(i,j,nThreads,tid,my_first,my_last,j_first, j_last,mtd2)
	{
		mtd2 = mtd_info_init(eccsize,oobsize,eccsize,eccbytes);

		tid = omp_get_thread_num();
		nThreads = omp_get_num_threads();
		my_first =   (   tid       * numpage ) / nThreads;
		my_last  =   (( ( tid + 1 ) * numpage ) / nThreads) - 1;
		j_first =   (   tid       * numerrpages ) / nThreads;
                j_last  =   (( ( tid + 1 ) * numerrpages ) / nThreads) - 1;
	
		printf("numerrpages: %d, j_first: %d , j_last: %d \n",numerrpages, j_first, j_last);	
		printf("my_first: %d , my_last: %d \n", my_first, my_last);

		for(i=my_first, j=j_first; i<= my_last && j<=j_last; i++){

			if(i== errindexes[j]){
				printf("tid: %d , error index: %d \n", tid, i);
				nandbch_test_prepare(mtd2,(void *)&correct_data[i * bufsize],(void *)&correct_ecc[i * eccsize],(void *)&error_data[i * bufsize],(void *)&error_ecc[i * eccsize],bufsize,t,1);
				j++;
			}else
				nandbch_test_prepare(mtd2,(void *)&correct_data[i * bufsize],(void *)&correct_ecc[i * eccsize],(void *)&error_data[i * bufsize],(void *)&error_ecc[i * eccsize],bufsize,t,0);
		
		}
	}*/
	
	gettimeofday(&tv1,NULL);	

	#pragma omp parallel private(i,nThreads,tid,my_first,my_last,err,mtd2)
	{
		mtd2 = mtd_info_init(eccsize,oobsize,eccsize,eccbytes);

		tid = omp_get_thread_num();
		nThreads = omp_get_num_threads();
		my_first =   (   tid       * numpage ) / nThreads;
		my_last  =   (( ( tid + 1 ) * numpage ) / nThreads) - 1;
		
		for(i=my_first; i<= my_last; i++){
			err = nand_bch_correct_data_SSE(mtd2,(void *)&error_data[i * bufsize],(void *)&error_ecc[i * eccsize],(void *)&correct_ecc[i * eccsize]);
			//if(err == t || err == -1)
				//printf("tid: %d, i: %d, err: %d \n", tid, i, err);
		}
		mtd_info_free(mtd2);
	}
	gettimeofday(&tv2,NULL);
	sec = (int) (tv2.tv_sec-tv1.tv_sec);
	usec = (int) (tv2.tv_usec-tv1.tv_usec);
	if (usec < 0){
			sec--;
			usec += 1000000;
	}
	printf("Total time for %d MB : %f sec\n",memsize * 2,sec+usec/1000000.0);

	//free the momory
	mtd_info_free(mtd);	
	free(error_data);
	free(error_ecc);
	free(correct_data);
	free(correct_ecc);
			
}
