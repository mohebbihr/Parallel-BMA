// BMAOpenMP-SSELess.cpp : Defines the entry point for the console application.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <emmintrin.h>
#include <assert.h>
#include <stdint.h>
#include <omp.h>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#define min(a,b) ((a) < (b) ? (a) : (b))
#define bufferSize 1024

int NUM_THREADS=12; //equal the number of cores of the hardware
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
// the number of iterations
//int numloop=0;


static inline __m128i muly(const __m128i a, const __m128i b)
{
#ifdef __SSE4_1__  // modern CPU - use SSE 4.1
    return _mm_mullo_epi32(a, b);
#else               // old CPU - use SSE 2
    __m128i tmp1 = _mm_mul_epu32(a,b); /* mul 2,0*/
    __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4)); /* mul 3,1 */
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
#endif
}

static inline __m128i _mm_bitcounts(__m128i a){
	
    __m128i tt,tt2;

    tt = _mm_srli_epi32( a, 1); //d = d - ((d >> 1) & 0x55555555);
    tt = _mm_and_si128(tt,_mm_set1_epi32(0x55555555));
    a = _mm_sub_epi32(a,tt);
    tt2 = _mm_and_si128(a, _mm_set1_epi32(0x33333333));    //d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
    tt = _mm_srli_epi32( a, 2);
    tt = _mm_and_si128(tt,_mm_set1_epi32(0x33333333));
    a = _mm_add_epi32(tt, tt2);
    tt = _mm_add_epi32(a, _mm_srli_epi32(a, 4)); //d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    tt = _mm_and_si128(tt,_mm_set1_epi32(0xF0F0F0F));
    tt = muly(tt,_mm_set1_epi32(0x1010101));
    a = _mm_srli_epi32( tt, 24);               
        
    return a;
}

static inline __m128i _mm_popcnt(const __m128i a)
{
        __m128i r;
        int * p1 = (int *) &a;
        int * p2 = (int *) &r;
        p2[0] = _mm_popcnt_u32(p1[0]);
        p2[1] = _mm_popcnt_u32(p1[1]);
        p2[2] = _mm_popcnt_u32(p1[2]);
        p2[3] = _mm_popcnt_u32(p1[3]);

        return r;
}

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
		printf("Usage: ./BMAOpenMP-SSE1D filename length -s NumberofSerialInputs -t NumberofThreads  -b debugLvl\n");
		printf("\tto generate random string: ./BMAOpenMP-SSE1D randGen length\n");
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
        bitRS = new unsigned**[nSerialInputs];
        bitC = new unsigned*[nSerialInputs];
        bitD = new unsigned*[nSerialInputs];
        bitTmp = new unsigned*[nSerialInputs];
        bitLenS = new unsigned[nSerialInputs];
        bitLenC = new unsigned[nSerialInputs];
	lenC = new unsigned[nSerialInputs];
	lenD = new unsigned[nSerialInputs];
	//lenS = new unsigned[nSerialInputs];
	n = new unsigned[nSerialInputs];
	m = new int[nSerialInputs];
	
	for(i=0; i<nSerialInputs; i++){
		bitLenS[i] = (lenS+31)/32;
		//bitLenS must be minimum 4 for applying SSE instructions
		if(bitLenS[i] <=4 ) bitLenS[i] =4;
		bitRS[i] = new unsigned*[32];
		for(j=0; j<32;j++)
			bitRS[i][j] = (unsigned*) aligned_malloc((bitLenS[i]+1) * sizeof(unsigned), 16);
		bitC[i] = (unsigned*)aligned_malloc(sizeof(unsigned)*bitLenS[i],16);
		bitD[i] = (unsigned*)aligned_malloc(sizeof(unsigned)*(bitLenS[i]+16),16);
		for(j=0; j<16; j++)
		  bitD[i][j] = 0;		  
		bitD[i]+=16;
		bitTmp[i] = (unsigned*)aligned_malloc(sizeof(unsigned)*bitLenS[i],16);
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


void bitBMASSE1DSerial(int input_index){
  
	int i,q,r,t, upperBound, wordCnt, shiftD, startC, word, bitPos;	
	int n3,n2,n1;
	unsigned d;
	int * qv;
	int * rv;
	int * c;	

	//sse pointers
	__m128i* bitDSSE = (__m128i*) bitD[input_index];
	__m128i* bitCSSE = (__m128i*) bitC[input_index];
	__m128i* bitTmpSSE = (__m128i*) bitTmp[input_index];	
	__m128i qSSE,lenCSSE, lenDSSE , bitLenCSSE, dSSE ;
	__m128i tt,tt2,rSSE;
	__m128i m27,c1SSE;
	
	// using sse to initialize the bitC, and bitD
	bitCSSE[0] = _mm_set_epi32(0,0,0,power2[31]);
	bitDSSE[0] = _mm_set_epi32(0,0,0,power2[31]);	
	
	for(i=1;i<(bitLenS[input_index]>>2);i++) {
		bitCSSE[i] = bitDSSE[i] = _mm_set1_epi32(0);				
	}
	t = (bitLenS[input_index]>>2)<<2;
	for(i=t; i<bitLenS[input_index]; i++){ 						
		bitCSSE[i] = bitDSSE[i] = _mm_set1_epi32(0);				
	}
	
	n[input_index] = lenC[input_index] = lenD[input_index] = 0;
	m[input_index] = -1;

	while (n[input_index]<lenS){	
	  //for debugging
	  //numloop++;	
	  
	  if(n[input_index] <=128){
	    // for the beginning of the process we use bit operations that is fast.
	    q= (lenS-1-n[input_index]) >> 5;
	    r = (lenS-1-n[input_index]) & ~bitMask[27];
	    bitLenC[input_index] = (lenC[input_index]+1+31)>>5;

	    d = 0;
	    for(i=0;i<bitLenC[input_index];i++) d ^= bitC[input_index][i] & bitRS[input_index][r][q+i];
	    d = d - ((d >> 1) & 0x55555555);
	    d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
	    d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	    if (d & power2[0]){
	      if (lenC[input_index]<=(n[input_index]>>1))
		for(i=0;i<bitLenC[input_index];i++) bitTmp[input_index][i] = bitC[input_index][i];
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
		for(i=0;i<bitLenC[input_index];i++) bitD[input_index][i] = bitTmp[input_index][i];
		lenD[input_index] = lenC[input_index];
		lenC[input_index] = n[input_index]+1-lenC[input_index];
		m[input_index] = n[input_index];
	      }
	    }
	    n[input_index]++;
	    
	  }else{
	    // when n is bigger then 128, we use SSE instructions.
		
		q= (lenS-1-n[input_index])>>5;
		r = (lenS-1-n[input_index]) & ~bitMask[27];	
												
		bitLenC[input_index] = (lenC[input_index]+1+31)>>5;
		d=0;		
		dSSE = _mm_set1_epi32(0);

		for(i=0;i<(bitLenC[input_index]>>2);i++){
			t = i<<2;
			dSSE = _mm_xor_si128(dSSE,_mm_and_si128(_mm_set_epi32(bitC[input_index][t+3],bitC[input_index][t+2],bitC[input_index][t+1],bitC[input_index][t]),
			_mm_set_epi32(bitRS[input_index][r][q+t+3],bitRS[input_index][r][q+t+2],bitRS[input_index][r][q+t+1],bitRS[input_index][r][q+t])));

		}
		c = (int *) &dSSE;
		d = (c[0] ^ c[1]) ^ (c[2] ^ c[3]);
		t = (bitLenC[input_index]>>2)<<2;
		for(i=t; i<bitLenC[input_index]; i++)
			d ^=  bitC[input_index][i] & bitRS[input_index][r][q+i];

		d = d - ((d >> 1) & 0x55555555);
		d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
		d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;

		if (d & power2[0]){							
			if (lenC[input_index]<=(n[input_index]>>1)){
				for(i=0;i<(bitLenC[input_index]>>2);i++) bitTmpSSE[i] = bitCSSE[i];
				t = (bitLenC[input_index]>>2)<<2;
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
					for(i=0;i<(bitLenC[input_index]>>2);i++) bitDSSE[i] = bitTmpSSE[i]; //bitD[i] = bitTmp[i];
					t = (bitLenC[input_index]>>2)<<2;
					for(i=t; i<bitLenC[input_index]; i++) bitD[input_index][i] = bitTmp[input_index][i];
					lenD[input_index] = lenC[input_index];
					lenC[input_index] = n[input_index]+1-lenC[input_index];
					m[input_index] = n[input_index];
			}
	
		  }	
		  n[input_index]++;		
	      }				  
		
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
	
	//!!!! uncommenting this area, makes the code produce wrong results. The reason is unknown!!!!
	/*if (debug){
	  printf("BMAOpenMP-SSELess: degree is %d for input: %d\n",lenC[input_index],input_index);
          bitPrint(bitC[input_index],lenC[input_index]+1);
	}*/		
	
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
				bitBMASSE1DSerial(i);
		  }

		  /*for(i=0; i<(nSerialInputs/nThreads); i++){
		     if(i*nThreads+tid <nSerialInputs && i*nThreads+tid>=0)
			bitBMASSE1DSerial(i*nThreads+tid); 
		  }
		 //printf("after first loop, tid:%d\n",tid);
		  j = (nSerialInputs/nThreads)*nThreads;
		  i=j;
		  //for(i=j; i<nSerialInputs; i++){
		     if(i+tid <nSerialInputs && i+tid>=0)
			bitBMASSE1DSerial(i+tid); 
		  //}*/
	  }

	  if (debug){
	  	for(i=0; i<nSerialInputs; i++){		
         		printf("BMAOpenMP-SSE1D: degree is %d for input: %d\n",lenC[i],i);
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
	  printf("BMAOpenMP-SSE1D: for %d inputs: %f sec\n",nSerialInputs,sec+usec/1000000.0);
	  //printf("numloop: %d\n", numloop);	 
	  fp = fopen("results.txt","w");
        fprintf(fp,"BMAOpenMP-SSE1D, result of executing for %d len, %d inputs, %d threads: %f sec\n",lenS,nSerialInputs,NUM_THREADS,sec+usec/1000000.0);

        fclose(fp);


	  //free the memory use
	  for(i=0; i<nSerialInputs; i++){
		for(j=0;j<32;j++)
			aligned_free(bitRS[i][j]);
		aligned_free(bitC[i]);
		aligned_free(bitD[i]);
		aligned_free(bitTmp[i]);
          }
	  return 0;
}

