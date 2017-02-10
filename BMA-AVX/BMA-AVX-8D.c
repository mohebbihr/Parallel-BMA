// BMA-AVX-8D.c : We predict 8 values for d using AVX instructions. In the best case scenario, the number of iterations would be reduced by 8.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <assert.h>
#include <stdint.h>


#define min(a,b) ((a) < (b) ? (a) : (b))
#define bufferSize 1024

char buffer[bufferSize];
int debug = 0, randGen = 0, lenS, m;
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
unsigned *bitRS[32], bitLenS, bitLenC;
unsigned* bitC;
unsigned* bitD;
unsigned* bitTmp;
//just for debugging
//int numloop=0;

inline __m256i _m256i_shiftright(const __m256i a , int num){
	// this function treat __m256 as integer
	int * p = (int * )&a;
	__m128i r0 = _mm_srli_epi32(_mm_set_epi32(p[3],p[2],p[1],p[0]),num);
	__m128i r1 = _mm_srli_epi32(_mm_set_epi32(p[7],p[6],p[5],p[4]),num);
	
	int * p0 = (int *)&r0;
	int * p1 = (int *)&r1;
	
	return _mm256_set_epi32(p1[3],p1[2],p1[1],p1[0],p0[3],p0[2],p0[1],p0[0]);
}

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

inline __m256i _m256i_bitcounts(const __m256i a){
        int * p = (int * )&a;
        __m128i r0 = _mm_set_epi32(p[3],p[2],p[1],p[0]);
        __m128i r1 = _mm_set_epi32(p[7],p[6],p[5],p[4]);
        __m128i tt,tt2;

        tt = _mm_srli_epi32( r0, 1); //d = d - ((d >> 1) & 0x55555555);
        tt = _mm_and_si128(tt,_mm_set1_epi32(0x55555555));
        r0 = _mm_sub_epi32(r0,tt);
        tt2 = _mm_and_si128(r0, _mm_set1_epi32(0x33333333));    //d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
        tt = _mm_srli_epi32( r0, 2);
        tt = _mm_and_si128(tt,_mm_set1_epi32(0x33333333));
        r0 = _mm_add_epi32(tt, tt2);
        tt = _mm_add_epi32(r0, _mm_srli_epi32(r0, 4)); //d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        tt = _mm_and_si128(tt,_mm_set1_epi32(0xF0F0F0F));
        tt = muly(tt,_mm_set1_epi32(0x1010101));
        r0 = _mm_srli_epi32( tt, 24);

        tt = _mm_srli_epi32( r1, 1); //d = d - ((d >> 1) & 0x55555555);
        tt = _mm_and_si128(tt,_mm_set1_epi32(0x55555555));
        r1 = _mm_sub_epi32(r1,tt);
        tt2 = _mm_and_si128(r1, _mm_set1_epi32(0x33333333));    //d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
        tt = _mm_srli_epi32( r1, 2);
        tt = _mm_and_si128(tt,_mm_set1_epi32(0x33333333));
        r1 = _mm_add_epi32(tt, tt2);
        tt = _mm_add_epi32(r1, _mm_srli_epi32(r1, 4)); //d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        tt = _mm_and_si128(tt,_mm_set1_epi32(0xF0F0F0F));
        tt = muly(tt,_mm_set1_epi32(0x1010101));
        r1 = _mm_srli_epi32( tt, 24);

        int * p0 = (int *)&r0;
        int * p1 = (int *)&r1;
        
        return _mm256_set_epi32(p1[3],p1[2],p1[1],p1[0],p0[3],p0[2],p0[1],p0[0]);
}

static inline __m256i _m256i_popcnt(const __m256i a)
{
        __m256i r;
        int * p1 = (int *) &a;
        int * p2 = (int *) &r;
        p2[0] = _mm_popcnt_u32(p1[0]);
        p2[1] = _mm_popcnt_u32(p1[1]);
        p2[2] = _mm_popcnt_u32(p1[2]);
        p2[3] = _mm_popcnt_u32(p1[3]);
        p2[4] = _mm_popcnt_u32(p1[4]);
        p2[5] = _mm_popcnt_u32(p1[5]);
        p2[6] = _mm_popcnt_u32(p1[6]);
        p2[7] = _mm_popcnt_u32(p1[7]);
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
	if (bitPos == -1){
		bitPos = 31;
		j++;
	}
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
		printf("Usage: ./BMA-AVX-LessIterations filename length -b debugLvl\n");
		printf("\tto generate random string: ./BMA-AVX-LessIterations randGen length\n");
		exit(1);
	}
	if (argc > 3){
		i = 3;
		while (i<argc){
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
	bitLenS = (lenS+31)/32;
	
	//bitLenS must be minimum 8 for applying AVX instructions
	if(bitLenS < 8 ) bitLenS =8;
		  
	for(i=0;i<32;i++)
	      bitRS[i] = (unsigned*) aligned_malloc((bitLenS+1) * sizeof(unsigned), 32);
	bitC = (unsigned*) aligned_malloc(bitLenS * sizeof(unsigned), 32);	
	bitD = (unsigned*) aligned_malloc((bitLenS + 32) * sizeof(unsigned), 32);		
	for(i=0; i<32; i++)
	  bitD[i] = 0;		  
	bitD+=32; 
	bitTmp = (unsigned*) aligned_malloc(bitLenS * sizeof(unsigned), 32);					

	j = 0;
	bitRS[0][j] = 0;
	bitPos = 31;
	for(i=0;i<lenS;i++){
	if (rS[i]) bitRS[0][j] |= power2[bitPos];
	bitPos--;
	if (bitPos == -1){
		bitPos = 31;
		j++;
		bitRS[0][j] = 0;
	}
	}
	bitRS[0][bitLenS] = 0;
	for(i=1;i<32;i++){
	for(j=0;j<bitLenS;j++)
		bitRS[i][j] = (bitRS[0][j] << i) |
	((bitRS[0][j+1] & bitMask[i]) >> (32-i));
	bitRS[i][bitLenS] = 0;
	}

}


void bitBMAVX(void){
  
	int i,q,r,t, upperBound, wordCnt, shiftD, startC, word, bitPos;	
	int n7,n6,n5,n4,n3,n2,n1;
	unsigned d;
	int * qv;
	int * rv;
	int * c;
	
	//AVX pointers
	__m256i * bitDSSE = (__m256i *) bitD;
	__m256i * bitCSSE = (__m256i *) bitC;	
	__m256i * bitTmpSSE = (__m256i *) bitTmp;	
	__m256i qSSE,lenCSSE, lenDSSE , bitLenCSSE, dSSE ;
	__m256i tt,tt2,rSSE;
	__m256i m27,c1SSE;
	
	// using AVX instruction to initialize the bitC, and bitD
	bitCSSE[0] = _mm256_set_epi32(0,0,0,0,0,0,0,power2[31]);
	bitDSSE[0] = _mm256_set_epi32(0,0,0,0,0,0,0,power2[31]);
	
	for(i=1;i<(bitLenS>>3);i++) {
		bitCSSE[i] = bitDSSE[i] = _mm256_setzero_si256();				
	}
	t = (bitLenS>>3)<<3;
	for(i=t; i<bitLenS; i++){ 						
		bitCSSE[i] = bitDSSE[i] = _mm256_setzero_si256();				
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
	    bitLenC = (lenC+1+31)>>5;

	    d = 0;
	    for(i=0;i<bitLenC;i++) d ^= bitC[i] & bitRS[r][q+i];
	    d = d - ((d >> 1) & 0x55555555);
	    d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
	    d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	    if (d & power2[0]){
	      if (lenC<=(n>>1))
		for(i=0;i<bitLenC;i++) bitTmp[i] = bitC[i];
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
		  bitC[startC+i] ^= ((bitD[i-1] & ~bitMask[32-shiftD]) << (32-shiftD))
		    | ((bitD[i] & bitMask[32-shiftD]) >> shiftD);
		else
		  bitC[startC+i] ^= bitD[i];
	      if (lenC<=(n>>1)){
		for(i=0;i<bitLenC;i++) bitD[i] = bitTmp[i];
		lenD = lenC;
		lenC = n+1-lenC;
		m = n;
	      }
	    }
	    n++;
	    
	  }else{	    
	    // when n is bigger then 128, we use AVX instructions.
	    if(n+7 <lenS){
			n7 = lenS-8-n;
		}else n7 = 0;
		
		if(n+6 <lenS){
			n6 = lenS-7-n;
		}else n6 = 0;
		
		if(n+5 <lenS){
			n5 = lenS-6-n;
		}else n5 = 0;
		
		if(n+4 <lenS){
			n4 = lenS-5-n;
		}else n4 = 0;
		
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
		tt = _mm256_set_epi32(n7,n6,n5,n4,n3, n2, n1, lenS-1-n);
		qSSE = _m256i_shiftright(tt,5);	

		//r = (lenS-1-n) & ~bitMask[27];	
		m27 = _mm256_set1_epi32(bitMask[27]);
		rSSE = _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(m27),_mm256_castsi256_ps(tt)));			
												
		bitLenC = (lenC+1+31)>>5;		
		dSSE = _mm256_setzero_si256();					

		qv = (int *) &qSSE;
		rv = (int *) &rSSE;

		for(i=0;i<bitLenC;i++){							
			tt = _mm256_set_epi32(bitRS[rv[7]][qv[7]+i], bitRS[rv[6]][qv[6]+i],bitRS[rv[5]][qv[5]+i],bitRS[rv[4]][qv[4]+i],
			bitRS[rv[3]][qv[3]+i],bitRS[rv[2]][qv[2]+i],bitRS[rv[1]][qv[1]+i],bitRS[rv[0]][qv[0]+i]);
            tt = _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(bitC[i])),_mm256_castsi256_ps(tt)));        
            dSSE = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(dSSE), _mm256_castsi256_ps(tt)));															
		}		

		//dSSE = _m256i_popcnt(dSSE);
		dSSE = _m256i_bitcounts(dSSE);		

		c1SSE = _mm256_castps_si256( _mm256_and_ps(_mm256_castsi256_ps(dSSE),_mm256_castsi256_ps(_mm256_set1_epi32(power2[0])))); // d & power2[0]
		c = (int *) &c1SSE;
		
		if(c[0]==0 && c[1]==0 && c[2]==0 && c[3]==0 ){ 
			n+=4;
			if(c[4]==0 && c[5]==0 && c[6]==0 && c[7]==0 ) n+=4;		
		}else{
		  if (c[0]){	

			if (lenC<=(n>>1)){						
				for(i=0;i<(bitLenC>>3);i++) bitTmpSSE[i] = bitCSSE[i];
				t = (bitLenC>>3)<<3;
				for(i=t; i<bitLenC; i++) bitTmp[i] = bitC[i];
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
				bitC[startC+i] ^= ((bitD[i-1] & ~bitMask[32-shiftD]) << (32-shiftD))
				| ((bitD[i] & bitMask[32-shiftD]) >> shiftD);
			else
				bitC[startC+i] ^= bitD[i];
			if (lenC<=(n>>1)){						
				for(i=0;i<(bitLenC>>3);i++) bitDSSE[i] = bitTmpSSE[i]; //bitD[i] = bitTmp[i];
				t = (bitLenC>>3)<<3;
				for(i=t; i<bitLenC; i++) bitD[i] = bitTmp[i];
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
	  if ((bitC[word] & power2[bitPos]) == 0) lenC--;
	  else break;
	  bitPos++;
	  if (bitPos == 32){
	    bitPos = 0;
	    word--;
	  }
	}
	
	if (debug){
	  printf("BMA-AVX-LessIterations: degree is %d\n",lenC);
	  if (debug > 1) bitPrint(bitC,lenC+1);
	}
	
	//free the memory use
	for(i=0;i<32;i++)
		aligned_free(bitRS[i]);
	aligned_free(bitC);
	aligned_free(bitD);
	aligned_free(bitTmp);
	
}

int main(int argc, char *argv[])
{		
	  struct timeval tv1, tv2;
	  int sec, usec;

	  init(argc,argv);
	  printf("input length %d\n",lenS);

	  gettimeofday(&tv1,NULL);
	
	  bitBMAVX(); 
	  
	  gettimeofday(&tv2,NULL);
	  sec = (int) (tv2.tv_sec-tv1.tv_sec);
	  usec = (int) (tv2.tv_usec-tv1.tv_usec);
	  if (usec < 0){
		sec--;
		usec += 1000000;
	  }
	  printf("BMA-AVX-LessIterations: %f sec\n",sec+usec/1000000.0);
	  //printf("numloop: %d\n", numloop);

	  return 0;
}

