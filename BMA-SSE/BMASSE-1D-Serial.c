// BMASSE-1D-Serial.c : This program computes the linear shift back register for n inputs. All the inputs have the same length and 
// the program do this computation in serial fashion. This program use BMASSE-1D implementation for each input. 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <emmintrin.h>
#include <assert.h>
#include <stdint.h>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#define min(a,b) ((a) < (b) ? (a) : (b))
#define bufferSize 1024

char buffer[bufferSize];
int debug = 0, randGen = 0, lenS, m;
int nSerialInputs =1; // the number of serial inputs
unsigned char *rS;
unsigned lenC, lenD, n, power2[32];
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

void* aligned_malloc(size_t size, size_t alignment) {
  
    uintptr_t r = (uintptr_t)malloc(size + --alignment + sizeof(uintptr_t));
    uintptr_t t = r + sizeof(uintptr_t);
    uintptr_t o =(t + alignment) & ~(uintptr_t)alignment;
    if (!r) return NULL;
    ((uintptr_t*)o)[-1] = r;
    return (void*)o;
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
		printf("Usage: ./BMASSE-1D-Serial filename -s NumberofSerialInputs length -b debugLvl\n");
		printf("\tto generate random string: ./BMASSE-LessSerial randGen length\n");
		exit(1);
	}
	if (argc > 3){
		i = 3;
		while (i<argc){
			if (!strcmp(argv[i],"-s")) sscanf(argv[i+1],"%d",&nSerialInputs);
			if (!strcmp(argv[i],"-b")) sscanf(argv[i+1],"%d",&debug);
			i += 2;
		}
	
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
	bitRS = (unsigned***) aligned_malloc(nSerialInputs * sizeof(unsigned**),16);
	bitC = (unsigned**) aligned_malloc(nSerialInputs * sizeof(unsigned*),16); 
	bitD = (unsigned**) aligned_malloc(nSerialInputs * sizeof(unsigned*),16); 
	bitTmp = (unsigned**) aligned_malloc(nSerialInputs * sizeof(unsigned*),16); 
	bitLenS = (unsigned*) aligned_malloc(nSerialInputs * sizeof(unsigned),16);    
	bitLenC = (unsigned*) aligned_malloc(nSerialInputs * sizeof(unsigned),16);            
	
	for(i=0; i<nSerialInputs; i++){
		bitLenS[i] = (lenS+31)/32;
		//bitLenS must be minimum 4 for applying SSE instructions
		if(bitLenS[i] <=4 ) bitLenS[i] =4;
		bitRS[i] = (unsigned**) aligned_malloc(32 * sizeof(unsigned*),16);
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


void bitBMASSESerial(int input_index){
  
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
	
	n = lenC = lenD = 0;
	m = -1;

	while (n<lenS){	
	  //for debugging
	  //numloop++;	
	  
	  if(n <=128){
	    // for the beginning of the process we use bit operations that is fast.
	    q = (lenS-1-n) >> 5;
	    r = (lenS-1-n) & ~bitMask[27];
	    bitLenC[input_index] = (lenC+1+31)>>5;

	    d = 0;
	    for(i=0;i<bitLenC[input_index];i++) d ^= bitC[input_index][i] & bitRS[input_index][r][q+i];
	    d = d - ((d >> 1) & 0x55555555);
	    d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
	    d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	    if (d & power2[0]){
	      if (lenC<=(n>>1))
		for(i=0;i<bitLenC[input_index];i++) bitTmp[input_index][i] = bitC[input_index][i];
	      upperBound = min(lenD+1,lenS+m-n);
	      startC = (n-m) >> 5;
	      shiftD = (n-m) & ~bitMask[27];
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
	      if (lenC<=(n>>1)){
		for(i=0;i<bitLenC[input_index];i++) bitD[input_index][i] = bitTmp[input_index][i];
		lenD = lenC;
		lenC = n+1-lenC;
		m = n;
	      }
	    }
	    n++;
	    
	  }else{	    
	    // when n is bigger then 128, we use SSE instructions.
	    
		if(n+3 <lenS){
			n3 = lenS-4-n;
		}else n3 = 0;

		if(n+2 <lenS){
			n2 = lenS-3-n; 
		}else n2 = 0;

		if(n+1 <lenS){
			n1 = lenS-2-n;
		}else n1 = 0;

		//q = (lenS-1-n) >> 5;	
		tt = _mm_set_epi32(n3, n2, n1, lenS-1-n);
		qSSE = _mm_srli_epi32(tt,5);

		//r = (lenS-1-n) & ~bitMask[27];	
		m27 = _mm_set1_epi32 (bitMask[27]);
		rSSE = _mm_andnot_si128(m27,tt);		
												
		bitLenC[input_index] = (lenC+1+31)>>5;		
		dSSE = _mm_set1_epi32(0);
		
		qv = (int *) &qSSE;
		rv = (int *) &rSSE;

		for(i=0;i<bitLenC[input_index];i++){							
			tt = _mm_set_epi32(bitRS[input_index][rv[3]][qv[3]+i], bitRS[input_index][rv[2]][qv[2]+i], bitRS[input_index][rv[1]][qv[1]+i], bitRS[input_index][rv[0]][qv[0]+i]);
			dSSE = _mm_xor_si128(dSSE,_mm_and_si128(_mm_set1_epi32(bitC[input_index][i]),tt));							
		}			
		
		dSSE = _mm_bitcounts(dSSE);
		//dSSE = _mm_popcnt(dSSE);
									
		c1SSE =  _mm_and_si128(dSSE,_mm_set1_epi32(power2[0])); // d & power2[0]
		c = (int *) &c1SSE;

		if(c[0]==0 && c[1]==0 && c[2]==0 && c[3]==0 ) n+=4;		
		else{
		  if (c[0]){							
		    if (lenC<=(n>>1)){						
				for(i=0;i<(bitLenC[input_index]>>2);i++) bitTmpSSE[i] = bitCSSE[i];
				t = (bitLenC[input_index]>>2)<<2;
				for(i=t; i<bitLenC[input_index]; i++) bitTmp[input_index][i] = bitC[input_index][i];
			}
			upperBound = min(lenD+1,lenS+m-n);
			startC = (n-m) >> 5;
			shiftD = (n-m) & ~bitMask[27];
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
			if (lenC<=(n>>1)){						
					for(i=0;i<(bitLenC[input_index]>>2);i++) bitDSSE[i] = bitTmpSSE[i]; //bitD[i] = bitTmp[i];
					t = (bitLenC[input_index]>>2)<<2;
					for(i=t; i<bitLenC[input_index]; i++) bitD[input_index][i] = bitTmp[input_index][i];
					lenD = lenC;
					lenC = n+1-lenC;
					m = n;
			}	
		  }	
		  n++;		
		}
		
	  }
					  		
	}
	

	word = (lenC+1) >> 5;
	bitPos = 32 - ((lenC+1) & ~bitMask[27]);
	if (bitPos == 32){
	  bitPos = 0;
	  word--;
	}

	while(1){
	  if ((bitC[input_index][word] & power2[bitPos]) == 0) lenC--;
	  else break;
	  bitPos++;
	  if (bitPos == 32){
	    bitPos = 0;
	    word--;
	  }
	}
	
	if (debug){
	  printf("BMASSE-LessSerial: degree is %d for input: %d\n",lenC,input_index);
      bitPrint(bitC[input_index],lenC+1);
	}		
	
}

int main(int argc, char *argv[])
{		
	  struct timeval tv1, tv2;
	  int i,j,sec, usec;
	  double t1;
	  FILE *fp;

	  init(argc,argv);
	  printf("input length %d\n",lenS);

	  gettimeofday(&tv1,NULL);
	  for(i=0; i<nSerialInputs; i++)
	  bitBMASSESerial(i); 
	  
	  gettimeofday(&tv2,NULL);
	  sec = (int) (tv2.tv_sec-tv1.tv_sec);
	  usec = (int) (tv2.tv_usec-tv1.tv_usec);
	  if (usec < 0){
		sec--;
		usec += 1000000;
	  }
          t1 = sec+usec/1000000.0;
	  printf("BMASSE-LessSerial: for %d inputs: %f sec\n",nSerialInputs,sec+usec/1000000.0);
	  //printf("numloop: %d\n", numloop);	 

	  //free the memory use
	  for(i=0; i<nSerialInputs; i++){
		for(j=0;j<32;j++)
			aligned_free(bitRS[i][j]);
		aligned_free(bitC[i]);
		aligned_free(bitD[i]);
		aligned_free(bitTmp[i]);
          }
	  
  	fp = fopen("results.txt","w");
  	fprintf(fp,"BMASSE-LessSerial, result of executing for len: %d, -s: %d \n",lenS,nSerialInputs);
  	fprintf(fp,"t1: %f\n",t1);
 
  	fclose(fp);

	return 0;
}

