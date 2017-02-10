// BMA-Xeon-OMP-nI.c
// This is the BMA implementation for Xeon Phi Co-processor which use openmp thread for each input sequence. 
// This version compiled natively for Xeon phi Co-processor. 

#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>
#define min(a,b) ((a) < (b) ? (a) : (b))
#define bufferSize 1024
#define vectorSize 16 // the number of integer values in one vector (512 bit)

int NUM_THREADS=240; //equal the number of cores of Xeon Phi
char buffer[bufferSize];
int debug = 0, randGen = 0, lenS, *m;
int nSerialInputs =1; // the number of serial inputs
unsigned char *rS;
unsigned *lenC, *lenD, *n, power2[32];
unsigned bitMask[32] = {
  0x00000000, 0x80000000, 0xC0000000, 0xE0000000,
  0xF0000000, 0xF8000000, 0xFC000000, 0xFE000000,
  0xFF000000, 0xFF800000, 0xFFC00000, 0xFFE00000,
  0xFFF00000, 0xFFF80000, 0xFFFC0000, 0xFFFE0000,
  0xFFFF0000, 0xFFFF8000, 0xFFFFC000, 0xFFFFE000,
  0xFFFFF000, 0xFFFFF800, 0xFFFFFC00, 0xFFFFFE00,
  0xFFFFFF00, 0xFFFFFF80, 0xFFFFFFC0, 0xFFFFFFE0,
  0xFFFFFFF0, 0xFFFFFFF8, 0xFFFFFFFC, 0xFFFFFFFE,
};
unsigned ***bitRS, *bitLenS, **bitC, *bitLenC, **bitD, **bitTmp;
unsigned **tt; 

void* aligned_malloc(size_t size, size_t alignment) {
  
    uintptr_t r = (uintptr_t)malloc(size + --alignment + sizeof(uintptr_t));
    uintptr_t t = r + sizeof(uintptr_t);
    uintptr_t o =(t + alignment) & ~(uintptr_t)alignment;
    if (!r) return NULL;
    ((uintptr_t*)o)[-1] = r;
    return (void*)o;
}

void aligned_free(void* p) {
    if (!p) return;
    free((void*)(((uintptr_t*)p)[-1]));
}

void bitPrint(unsigned *bitX, int length){
  int i, j, bitPos, tmp;

  bitPos = 31;
  j = 0;
  for(i=0;i<length;i++){
    tmp = bitX[j] & power2[bitPos];
    printf("%1d ",tmp?1:0);
    bitPos--;
    if (bitPos == -1) bitPos = 31, j++;
  }
  printf("\n");
}

void init(int argc, char *argv[])
{
	int i, j, bitPos;
	FILE *fp = NULL;
	char* token;	
	unsigned short randData[3];
	struct timeval tv;	

	if (argc < 3){
		printf("Usage: ./BMA-Xeon-OMP-nI filename length -s NumberofSerialInputs -t NumberofThreads  -b debugLvl\n");
		printf("\tto generate random string: ./BMA-Xeon-OMP-nI randGen length\n");
		exit(1);
	}
	if (argc > 3){
		i = 3;
		while (i<argc){
			if (!strcmp(argv[i],"-s")) sscanf(argv[i+1],"%d",&nSerialInputs);
			if (!strcmp(argv[i],"-t")) sscanf(argv[i+1],"%d",&NUM_THREADS);
			if (!strcmp(argv[i],"-b")) sscanf(argv[i+1],"%d",&debug);
			i += 2;
		}
	
	}
	if(NUM_THREADS <=0){
		printf("number of threads must be greater than zero\n");
		exit(1);
	}	
	if (strcmp(argv[1],"randGen") == 0) randGen = 1;
	else{
		randGen = 0;
		fp = fopen(argv[1],"r");
		if (!fp){
			printf("%s doesn't exist\n",argv[1]);
			exit(1);
		}
	}
	sscanf(argv[2],"%d",&lenS);
	if (lenS <= 0){
		printf("positive length needed\n");
		exit(1);
	}	
		
	rS = (unsigned char*)malloc(sizeof(unsigned char)*lenS);
	if (randGen){
		gettimeofday(&tv,NULL);
		randData[0] = (unsigned short) tv.tv_usec;
		randData[1] = (unsigned short) tv.tv_sec;
		randData[2] = (unsigned short) (tv.tv_sec >> 16);
		rS[lenS-1] = 1;
		for(i=2;i<=lenS;i++) rS[lenS-i] = (erand48(randData) > 0.5) ? 1 : 0;
	}
	else{
		i = 0;
		while (fgets(buffer,bufferSize,fp) && i<lenS){
			token = strtok(buffer," ");
			while (token){
				rS[lenS-i-1] = atoi(token);
				i++;
				token = strtok(NULL," ");
			}
		}
		fclose(fp);
		if (i != lenS){
			printf("file has only %d bits\n",i);
			exit(1);
		}
	}

	power2[0] = 1;
	for(i=1;i<32;i++) power2[i] = 2*power2[i-1];
	/* we have to have nSerialInputs array's for n serial inputs*/        
	bitRS = (unsigned***) aligned_malloc(nSerialInputs * sizeof(unsigned**),64);      
	bitC = (unsigned**) aligned_malloc(nSerialInputs * sizeof(unsigned*),64);    
	bitD = (unsigned**) aligned_malloc(nSerialInputs * sizeof(unsigned*),64);    
	bitTmp = (unsigned**) aligned_malloc(nSerialInputs * sizeof(unsigned*),64);        
	bitLenS = (unsigned*) aligned_malloc(nSerialInputs * sizeof(unsigned),64);    
	bitLenC = (unsigned*) aligned_malloc(nSerialInputs * sizeof(unsigned),64);	
	lenC = (unsigned*) aligned_malloc(nSerialInputs * sizeof(unsigned),64);	
	lenD = (unsigned*) aligned_malloc(nSerialInputs * sizeof(unsigned),64);		
	n = (unsigned*) aligned_malloc(nSerialInputs * sizeof(unsigned),64);	
	m = (unsigned*) aligned_malloc(nSerialInputs * sizeof(unsigned),64);	
	tt = (unsigned**) aligned_malloc(nSerialInputs * sizeof(unsigned*),64);	
	
	for(i=0; i<nSerialInputs; i++){
		bitLenS[i] = (lenS+31)/32;
		//bitLenS must be minimum 8 for applying AVX instructions
		if(bitLenS[i] <=8 ) bitLenS[i] =8;		
		bitRS[i] = (unsigned**) aligned_malloc(32 * sizeof(unsigned*),64);
		for(j=0; j<32;j++)
			bitRS[i][j] = (unsigned*) aligned_malloc((bitLenS[i]+1) * sizeof(unsigned), 64);
				
		tt[i] = (unsigned*) aligned_malloc(vectorSize * sizeof(unsigned),64);		
		bitC[i] = (unsigned*)aligned_malloc(sizeof(unsigned)*bitLenS[i],64);
		bitD[i] = (unsigned*)aligned_malloc(sizeof(unsigned)*(bitLenS[i]+64),64);
		for(j=0; j<64; j++)
		  bitD[i][j] = 0;		  
		bitD[i]+=64;
		bitTmp[i] = (unsigned*)aligned_malloc(sizeof(unsigned)*bitLenS[i],64);
		j = 0;
		bitRS[i][0][j] = 0;
		bitPos = 31;
		int ix;
		for(ix=0;ix<lenS;ix++){
			if (rS[ix]) bitRS[i][0][j] |= power2[bitPos];
			bitPos--;
			if (bitPos == -1){
			  bitPos = 31;
			  j++;
			  bitRS[i][0][j] = 0;
			}
		}
		bitRS[i][0][bitLenS[i]] = 0;
		for(ix=1;ix<32;ix++){
			for(j=0;j<bitLenS[i];j++)
			  bitRS[i][ix][j] = (bitRS[i][0][j] << ix) |
			((bitRS[i][0][j+1] & bitMask[ix]) >> (32-ix));
			bitRS[i][ix][bitLenS[i]] = 0;
		}
	}

}


void BMA_Xeon_OMP(int input_index){
  
	int i,j,q,r,t, upperBound, wordCnt, shiftD, startC, word, bitPos;	
	int n3,n2,n1;
	unsigned d;
	int * qv;
	int * rv;
	int * c;	
	
	__m512i* bitDAVX = (__m512i*) bitD[input_index];
	__m512i* bitCAVX = (__m512i*) bitC[input_index];
	__m512i* bitTmpAVX = (__m512i*) bitTmp[input_index];	
	__m512i dAVX;
	
	// initialize the array's				
	for(i=0; i< (bitLenS[input_index]>>4); i++){
		bitDAVX[i] = _mm512_xor_epi32(bitDAVX[i], bitDAVX[i]);
        bitCAVX[i] = _mm512_xor_epi32(bitCAVX[i], bitCAVX[i]);			
	}
	t = (bitLenS[input_index]>>4)<<4;
	for(i=t; i<bitLenS[input_index]; i++) bitD[input_index][i] = bitC[input_index][i] = 0;			

	bitD[input_index][0] = bitC[input_index][0] = power2[31];		
	n[input_index] = 0; 
	lenC[input_index] = 0;
	lenD[input_index] = 0;
	m[input_index] = -1;	

	while (n[input_index]<lenS){		  		  	  
		
		q = (lenS-1-n[input_index])>>5;
		r = (lenS-1-n[input_index]) & ~bitMask[27];
		bitLenC[input_index] = (lenC[input_index]+1+31)>>5;
		d=0;		
		dAVX = _mm512_xor_epi32(dAVX,dAVX);		

		for(i=0;i<(bitLenC[input_index]>>4);i++){ // d computation			
			t = i<<4;
			for(j=0; j<vectorSize; j++){
				tt[input_index][j] = bitRS[input_index][r][q + t + j];				
			}
						
			dAVX = _mm512_xor_epi32(dAVX,_mm512_and_epi32(_mm512_load_epi32((void *)(bitC[input_index] + t)),_mm512_load_epi32((void *)(tt[input_index]))));							
		}
					
		int * c = (int *) &dAVX;
		d = c[0] ^ c[1] ^ c[2] ^ c[3] ^ c[4] ^ c[5] ^ c[6] ^ c[7] ^ c[8] ^ c[9] ^ c[10] ^ c[11] ^ c[12] ^ c[13] ^ c[14] ^ c[15];
		
		t = (bitLenC[input_index]>>4)<<4;
		for(i=t; i<bitLenC[input_index]; i++)
		d ^=  bitC[input_index][i] & bitRS[input_index][r][q+i];
		
		d = d - ((d >> 1) & 0x55555555);
		d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
		d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;

		if (d & power2[0]){							
			if (lenC[input_index]<=(n[input_index]>>1)){
				
				for(i=0;i<(bitLenC[input_index]>>4);i++){	
					for(i=0;i<(bitLenC[input_index]>>4);i++) bitTmpAVX[i] = bitCAVX[i];
					t = (bitLenC[input_index]>>4)<<4;
					for(i=t; i<bitLenC[input_index]; i++) bitTmp[input_index][i] = bitC[input_index][i];									
				} 
				t = (bitLenC[input_index]>>4)<<4;
				for(i=t; i<bitLenC[input_index]; i++) bitTmp[input_index][i] = bitC[input_index][i];
										
			}
			upperBound = min(lenD[input_index]+1,lenS+m[input_index]-n[input_index]);
			startC = (n[input_index]-m[input_index]) >> 5;
			shiftD = (n[input_index]-m[input_index]) & ~bitMask[27];
			wordCnt = 0;
			if (shiftD){
				upperBound -= (32-shiftD);
				wordCnt++;
			}
			wordCnt += (upperBound+31) >> 5;
			for(i=0;i<wordCnt;i++)
				if (shiftD)
					bitC[input_index][startC+i] ^= ((bitD[input_index][i-1] & ~bitMask[32-shiftD]) << (32-shiftD))
					| ((bitD[input_index][i] & bitMask[32-shiftD]) >> shiftD);
				else
					bitC[input_index][startC+i] ^= bitD[input_index][i];
			
			if (lenC[input_index]<=(n[input_index]>>1)){
				for(i=0;i<(bitLenC[input_index]>>4);i++){
					for(i=0;i<(bitLenC[input_index]>>4);i++)  bitDAVX[i] = bitTmpAVX[i];
					t = (bitLenC[input_index]>>4)<<4;
					for(i=t; i<bitLenC[input_index]; i++) bitD[input_index][i] = bitTmp[input_index][i];   						
				} 
				t = (bitLenC[input_index]>>4)<<4;
				for(i=t; i<bitLenC[input_index]; i++) bitD[input_index][i] = bitTmp[input_index][i];				

				lenD[input_index] = lenC[input_index];
				lenC[input_index] = n[input_index]+1-lenC[input_index];
				m[input_index] = n[input_index];
			}
	
		  }	
		  n[input_index]++;			      				 
		
	}
	

	word = (lenC[input_index]+1) >> 5;
	bitPos = 32 - ((lenC[input_index]+1) & ~bitMask[27]);
	if (bitPos == 32){
	  bitPos = 0;
	  word--;
	}

	while(1){
	  if ((bitC[input_index][word] & power2[bitPos]) == 0) lenC[input_index]--;
	  else break;
	  bitPos++;
	  if (bitPos == 32){
	    bitPos = 0;
	    word--;
	  }
	}
		
}

int main(int argc, char *argv[])
{		
	  struct timeval tv1, tv2;
	  int i,j,nThreads,tid,sec, usec,my_first,my_last;
	  FILE *fp;
	
	  init(argc,argv);
	  if(nSerialInputs < NUM_THREADS)
		NUM_THREADS = nSerialInputs;
	  omp_set_num_threads(NUM_THREADS);
	  printf("input length %d - number of threads: %d\n",lenS,NUM_THREADS);
	  gettimeofday(&tv1,NULL);

	  #pragma omp parallel private(i,nThreads,tid,my_first,my_last)
	  {
		  tid = omp_get_thread_num();
		  nThreads = omp_get_num_threads();
		  
		  my_first =   (   tid       * nSerialInputs ) / nThreads;
		  my_last  =   ( ( tid + 1 ) * nSerialInputs ) / nThreads - 1;
		  
		  for(i=my_first; i<= my_last; i++){
				BMA_Xeon_OMP(i);
		  }		
	  } 

	  if (debug){
	  	for(i=0; i<nSerialInputs; i++){		
         		printf("BMA-Xeon-OMP-nI: degree is %d for input: %d\n",lenC[i],i);
         		bitPrint(bitC[i],lenC[i]+1);
       		 }
	  }

	  gettimeofday(&tv2,NULL);
	  sec = (int) (tv2.tv_sec-tv1.tv_sec);
	  usec = (int) (tv2.tv_usec-tv1.tv_usec);
	  if (usec < 0){
		sec--;
		usec += 1000000;
	  }
	  printf("BMA-Xeon-OMP-nI: for %d inputs: %f sec\n",nSerialInputs,sec+usec/1000000.0);	 
	  //  fp = fopen("results.txt","w");
      //fprintf(fp,"BMA-Xeon-OMP-nI, result of executing for %d len, %d inputs, %d threads: %f sec\n",lenS,nSerialInputs,NUM_THREADS,sec+usec/1000000.0);

      //fclose(fp);

	  //free the memory use
	  for(i=0; i<nSerialInputs; i++){
		for(j=0;j<32;j++)
			aligned_free(bitRS[i][j]);
		aligned_free(bitC[i]);
		aligned_free(bitD[i]);
		aligned_free(bitTmp[i]);
        }
	  
		aligned_free(tt);	
		aligned_free(n);
		aligned_free(m);
		aligned_free(bitC);
	  return 0;
}




