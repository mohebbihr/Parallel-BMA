#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>

#define min(a,b) ((a) < (b) ? (a) : (b))
#define bufferSize 1024
#define NUMTHREAD 256
#define NUMBLOCK 64

char buffer[bufferSize];
// why deviceNum is 1 for default, it should be zero!!!!
int deviceNum = 1, debug = 0, randGen = 0, lenS, *m;
int nConInputStreams =1; // the number of concurrent streams of input
int nStream =4; //the number of streams
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

/* the way of defining these array must change */
unsigned ***bitRS, *bitLenS, **bitC, *bitLenC, **bitD, **bitTmp;
unsigned ***gpuBitRS, **gpuBitC, **gpuBitD, **gpuBitTmp, **gpuParity;
cudaStream_t *streams; // streams

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

void init(int argc, char* argv[]){
  int i, j, bitPos;
  FILE *fp = NULL;
  char* token;
  unsigned short randData[3];
  struct timeval tv;

  /* add nother input named (nConInputStreams) that determines the number of input streams */  
  if (argc < 3){
    printf("Usage: ./gpuBitStream filename length -c NumberofInputStreams -ns NumberofStreams -b debugLvl -d deviceNum\n");
    printf("\tto generate random string: ./gpuBitStream randGen length\n");
    exit(1);
  }
  if (argc > 3){
    i = 3;
    while (i<argc){
      if (!strcmp(argv[i],"-c")) sscanf(argv[i+1],"%d",&nConInputStreams);
	  if (!strcmp(argv[i],"-ns")) sscanf(argv[i+1],"%d",&nStream);
	  if (!strcmp(argv[i],"-b")) sscanf(argv[i+1],"%d",&debug);
      if (!strcmp(argv[i],"-d")) sscanf(argv[i+1],"%d",&deviceNum);	  
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
  
  /* we have to have nConInputStreams array's for n streams*/
  bitRS = new unsigned**[nConInputStreams];
  bitC = new unsigned*[nConInputStreams];
  bitD = new unsigned*[nConInputStreams];
  bitTmp = new unsigned*[nConInputStreams];
  bitLenS = new unsigned[nConInputStreams];
  bitLenC = new unsigned[nConInputStreams];
  
  for(i=0; i<nConInputStreams; i++){
	// bitLenS is same for all streams for now... later might change
	bitLenS[i] = (lenS+31)/32;
	bitRS[i] = new unsigned*[32];
	for(j=0; j<32;j++)
		bitRS[i][j] = (unsigned*)malloc(sizeof(unsigned)*(bitLenS[i]+1));
	bitC[i] = (unsigned*)malloc(sizeof(unsigned)*bitLenS[i]);
	bitD[i] = (unsigned*)malloc(sizeof(unsigned)*(bitLenS[i]+1));
	bitD[i][0] = 0;
	bitD[i]++;
	bitTmp[i] = (unsigned*)malloc(sizeof(unsigned)*bitLenS[i]);
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

void initGPU(void){
  int i,j;
  int num_devices=0;
  cudaGetDeviceCount(&num_devices);
  // check if the command-line chosen device ID is within range, exit if not
  if( deviceNum >= num_devices )
  {
	printf("choose device ID between 0 and %d\n", num_devices-1);
	exit(1);
  }
  cudaSetDevice(deviceNum);
  cudaDeviceProp deviceProp;	
  cudaGetDeviceProperties(&deviceProp, deviceNum);
  printf("> Device name : %s\n", deviceProp.name );
  printf("> CUDA Capable SM %d.%d hardware with %d multi-processors\n", deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount); 
  
  // allocate and initialize an array of stream handles
  streams = (cudaStream_t*) malloc(nStream * sizeof(cudaStream_t));
  for(i=0; i<nStream; i++)
	cudaStreamCreate(&(streams[i])); 
	
  /* we have to change the memory allocation for gpu as well as we done for cpu */
  gpuBitRS = new unsigned**[nConInputStreams];
  gpuBitC = new unsigned*[nConInputStreams];
  gpuBitD = new unsigned*[nConInputStreams];
  gpuBitTmp = new unsigned*[nConInputStreams];
  gpuParity = new unsigned*[nConInputStreams];
  
  for(j=0; j<nConInputStreams; j++){	
	gpuBitRS[j] = new unsigned*[32];
	for(i=0;i<32;i++){
		cudaMalloc((void**)&gpuBitRS[j][i],sizeof(unsigned)*(bitLenS[j]+1));
		cudaMemcpy(gpuBitRS[j][i],bitRS[j][i],sizeof(unsigned)*(bitLenS[j]+1),cudaMemcpyHostToDevice);
	}
	cudaMalloc((void**)&gpuBitC[j],sizeof(unsigned)*bitLenS[j]);
	cudaMalloc((void**)&gpuBitD[j],sizeof(unsigned)*(bitLenS[j]+1));
	cudaMalloc((void**)&gpuBitTmp[j],sizeof(unsigned)*bitLenS[j]);
	cudaMalloc((void**)&gpuParity[j],sizeof(unsigned)*NUMBLOCK);
  }
  
}

__global__ void kernel1(unsigned* rS, unsigned* C, unsigned* parity, int lenC){
  __shared__ unsigned sParity[NUMTHREAD];
  int myC = blockIdx.x*blockDim.x + threadIdx.x;

  sParity[threadIdx.x] = 0;
  if (myC > lenC) return;
  while (myC <= lenC){
    sParity[threadIdx.x] ^= C[myC] & rS[myC];
    myC += NUMTHREAD*NUMBLOCK;
  }
  __syncthreads();
  if (NUMTHREAD >= 1024){
    if (threadIdx.x < 512)
      sParity[threadIdx.x] ^= sParity[threadIdx.x+512];
    __syncthreads();
  }
  if (NUMTHREAD >= 512){
    if (threadIdx.x < 256)
      sParity[threadIdx.x] ^= sParity[threadIdx.x+256];
    __syncthreads();
  }
  if (NUMTHREAD >= 256){
    if (threadIdx.x < 128)
      sParity[threadIdx.x] ^= sParity[threadIdx.x+128];
    __syncthreads();
  }
  if (NUMTHREAD >= 128){
    if (threadIdx.x < 64)
      sParity[threadIdx.x] ^= sParity[threadIdx.x+64];
    __syncthreads();
  }
  if (threadIdx.x < 32){
    volatile unsigned *tmem = sParity;
    if (NUMTHREAD >= 64)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+32];
    if (NUMTHREAD >= 32)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+16];
    if (NUMTHREAD >= 16)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+8];
    if (NUMTHREAD >= 8)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+4];
    if (NUMTHREAD >= 4)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+2];
    if (NUMTHREAD >= 2)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+1];
  }
  if (threadIdx.x == 0) parity[blockIdx.x] = sParity[0];
}

__global__ void kernel2(unsigned* parity, int num){
  __shared__ unsigned sParity[NUMBLOCK];

  sParity[threadIdx.x] = (threadIdx.x < num) ? parity[threadIdx.x] : 0;
  __syncthreads();
  if (NUMBLOCK >= 1024){
    if (threadIdx.x < 512)
      sParity[threadIdx.x] ^= sParity[threadIdx.x+512];
    __syncthreads();
  }
  if (NUMBLOCK >= 512){
    if (threadIdx.x < 256)
      sParity[threadIdx.x] ^= sParity[threadIdx.x+256];
    __syncthreads();
  }
  if (NUMBLOCK >= 256){
    if (threadIdx.x < 128)
      sParity[threadIdx.x] ^= sParity[threadIdx.x+128];
    __syncthreads();
  }
  if (NUMBLOCK >= 128){
    if (threadIdx.x < 64)
      sParity[threadIdx.x] ^= sParity[threadIdx.x+64];
    __syncthreads();
  }
  if (threadIdx.x < 32){
    volatile unsigned *tmem = sParity;
    if (NUMBLOCK >= 64)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+32];
    if (NUMBLOCK >= 32)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+16];
    if (NUMBLOCK >= 16)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+8];
    if (NUMBLOCK >= 8)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+4];
    if (NUMBLOCK >= 4)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+2];
    if (NUMBLOCK >= 2)
      tmem[threadIdx.x] ^= tmem[threadIdx.x+1];
  }
  if (threadIdx.x == 0) parity[0] = sParity[0];
}

__global__ void
kernel3(unsigned* C, unsigned *D, int shiftD, unsigned mask, int num){
  int myIndex = blockIdx.x*blockDim.x + threadIdx.x;

  while (myIndex < num){
    C[myIndex] ^= ((D[myIndex-1] & ~mask) << (32-shiftD)) |
      ((D[myIndex] & mask) >> shiftD);
    myIndex += NUMTHREAD*NUMBLOCK;
  }
}

__global__ void kernel4(unsigned* C, unsigned* D, int num){
  int myIndex = blockIdx.x*blockDim.x + threadIdx.x;

  while (myIndex < num){
    C[myIndex] ^= D[myIndex];
    myIndex += NUMTHREAD*NUMBLOCK;
  }
}

void gpuBitStream(void){
  int i,j,t,index, *numBlock, *q, *r, *upperBound, *wordCnt, *shiftD, *startC, word, bitPos;
  unsigned *d;
  bool cond = true;
  
  d = new unsigned[nConInputStreams];
  n = new unsigned[nConInputStreams];
  lenC = new unsigned[nConInputStreams];
  lenD = new unsigned[nConInputStreams];
  m = new int[nConInputStreams];  
  q = new int[nConInputStreams];
  r = new int[nConInputStreams];
  upperBound = new int[nConInputStreams];
  numBlock = new int[nConInputStreams];
  startC = new int[nConInputStreams];
  shiftD = new int[nConInputStreams];
  wordCnt = new int[nConInputStreams];    
  
  for(i=0; i<nConInputStreams; i++){
	bitD[i][0] = bitC[i][0] = power2[31];
	for(j=1;j<bitLenS[i];j++) bitD[i][j] = bitC[i][j] = 0;			
	n[i] = lenC[i] = lenD[i] = 0;
	m[i] = -1;
	d[i] = 0;
  }   
   
  for(i=0;i<(nConInputStreams/nStream);i++){ 
	cond = true;
    // for now all the inputs have the same length (lenS)	
	for(j=0; j<nStream; j++){
		index = (i*nStream)+j;
		cudaMemcpyAsync(gpuBitC[index],bitC[index],sizeof(int)*bitLenS[index],cudaMemcpyHostToDevice,streams[j]);
		cudaMemcpyAsync(gpuBitD[index],bitD[index]-1,sizeof(int)*(bitLenS[index]+1),cudaMemcpyHostToDevice,streams[j]);
		gpuBitD[index]++;
	}
	
	while (cond == true){
		
		for(j=0; j<nStream; j++){
			// this part is serial in host CPU
			index = (i*nStream)+j;
			if(n[index]<lenS-1)
				cond = cond || true;
			else 
				cond = cond && false;
			
			q[index] = (lenS-1-n[index]) >> 5;
			r[index] = (lenS-1-n[index]) & ~bitMask[27];
			bitLenC[index] = (lenC[index]+1+31)>>5;
			numBlock[index] = (bitLenC[index]+NUMTHREAD-1)/NUMTHREAD * nStream;
			numBlock[index] = min(numBlock[index],NUMBLOCK);
		}											
		
		// executing the kernel1 for each stream		
		for(j=0; j<nStream; j++){
			index = (i*nStream)+j;
			kernel1<<<numBlock[index],NUMTHREAD, 0, streams[j]>>>(gpuBitRS[index][r[index]]+q[index],gpuBitC[index],gpuParity[index],bitLenC[index]);			
		}				
		// executing the kernel2 for each stream		
		for(j=0; j<nStream; j++){
			index = (i*nStream)+j;
			kernel2<<<1,NUMBLOCK, 0, streams[j]>>>(gpuParity[index],numBlock[index]);			
		}				
		for(j=0; j<nStream; j++){
			index = (i*nStream)+j;
			cudaMemcpyAsync(&d[index],gpuParity[index],sizeof(unsigned),cudaMemcpyDeviceToHost, streams[j]);			
		}		
		for(j=0; j<nStream; j++){
			index = (i*nStream)+j;
			d[index] = d[index]  - ((d[index] >> 1) & 0x55555555);
			d[index] = (d[index] & 0x33333333) + ((d[index] >> 2) & 0x33333333);
			d[index] = (((d[index] + (d[index] >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
			
			if (d[index] & power2[0]){
				if (lenC[index]<=(n[index]>>1))
					cudaMemcpyAsync(gpuBitTmp[index],gpuBitC[index],sizeof(unsigned)*bitLenC[index],cudaMemcpyDeviceToDevice, streams[j]);					
				upperBound[index] = min(lenD[index]+1,lenS+m[index]-n[index]);
				startC[index] = (n[index]-m[index]) >> 5;
				shiftD[index] = (n[index]-m[index]) & ~bitMask[27];
				wordCnt[index] = 0;
				if (shiftD[index]){
					upperBound[index] -= (32-shiftD[index]);
					wordCnt[index]++;
				}
				wordCnt[index] += (upperBound[index]+31) >> 5;
				numBlock[index] = (wordCnt[index]+NUMTHREAD-1)/NUMTHREAD * nStream;
				numBlock[index] = min(numBlock[index],NUMBLOCK);
			}
		}
		
		for(j=0; j<nStream; j++){
			index = (i*nStream)+j;
			if (d[index] & power2[0]){
				if (shiftD[index])
					kernel3<<<numBlock[index],NUMTHREAD, 0, streams[j]>>>(gpuBitC[index]+startC[index],gpuBitD[index],shiftD[index],bitMask[32-shiftD[index]],wordCnt[index]);					
				else
					kernel4<<<numBlock[index],NUMTHREAD, 0, streams[j]>>>(gpuBitC[index]+startC[index],gpuBitD[index],wordCnt[index]);					
			}
		}		
		for(j=0; j<nStream; j++){
			index = (i*nStream)+j;
			if (d[index] & power2[0]){
				if (lenC[index]<=(n[index]>>1)){
					cudaMemcpyAsync(gpuBitD[index],gpuBitTmp[index],sizeof(unsigned)*bitLenC[index],cudaMemcpyDeviceToDevice,streams[j]);					
					lenD[index]= lenC[index];
					lenC[index] = n[index]+1-lenC[index];
					m[index] = n[index];
				}	
			}			
			n[index]++;			
		}		
	}
	
	for(j=0; j<nStream; j++){
		index = (i*nStream)+j;
		cudaMemcpyAsync(bitC[index],gpuBitC[index],sizeof(int)*bitLenS[index],cudaMemcpyDeviceToHost,streams[j]);			
	}
	
  }
  t = (nConInputStreams/nStream)*nStream;
  for(i=t; i<nConInputStreams; i++){
	// the remaining ...	
	for(j=0; j<(nConInputStreams - t); j++){
		index = i;
		cudaMemcpyAsync(gpuBitC[index],bitC[index],sizeof(int)*bitLenS[index],cudaMemcpyHostToDevice,streams[j]);
		cudaMemcpyAsync(gpuBitD[index],bitD[index]-1,sizeof(int)*(bitLenS[index]+1),cudaMemcpyHostToDevice,streams[j]);
		gpuBitD[index]++;
	}
	
	while (cond == true){		
		for(j=0; j<(nConInputStreams - t); j++){
			// this part is serial in host CPU
			index = i;
			if(n[index]<lenS-1)
				cond = cond || true;
			else 
				cond = cond && false;
			
			q[index] = (lenS-1-n[index]) >> 5;
			r[index] = (lenS-1-n[index]) & ~bitMask[27];
			bitLenC[index] = (lenC[index]+1+31)>>5;
			numBlock[index] = (bitLenC[index]+NUMTHREAD-1)/NUMTHREAD * nStream;
			numBlock[index] = min(numBlock[index],NUMBLOCK);
		}											
		
		// executing the kernel1 for each stream		
		for(j=0; j<(nConInputStreams - t); j++){
			index = i;
			kernel1<<<numBlock[index],NUMTHREAD, 0, streams[j]>>>(gpuBitRS[index][r[index]]+q[index],gpuBitC[index],gpuParity[index],bitLenC[index]);			
		}				
		// executing the kernel2 for each stream		
		for(j=0; j<(nConInputStreams - t); j++){
			index = i;
			kernel2<<<1,NUMBLOCK, 0, streams[j]>>>(gpuParity[index],numBlock[index]);			
		}				
		for(j=0; j<(nConInputStreams - t); j++){
			index = i;
			cudaMemcpyAsync(&d[index],gpuParity[index],sizeof(unsigned),cudaMemcpyDeviceToHost, streams[j]);			
		}		
		for(j=0; j<(nConInputStreams - t); j++){
			index = i;
			d[index] = d[index]  - ((d[index] >> 1) & 0x55555555);
			d[index] = (d[index] & 0x33333333) + ((d[index] >> 2) & 0x33333333);
			d[index] = (((d[index] + (d[index] >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
			
			if (d[index] & power2[0]){
				if (lenC[index]<=(n[index]>>1))
					cudaMemcpyAsync(gpuBitTmp[index],gpuBitC[index],sizeof(unsigned)*bitLenC[index],cudaMemcpyDeviceToDevice, streams[j]);					
				upperBound[index] = min(lenD[index]+1,lenS+m[index]-n[index]);
				startC[index] = (n[index]-m[index]) >> 5;
				shiftD[index] = (n[index]-m[index]) & ~bitMask[27];
				wordCnt[index] = 0;
				if (shiftD[index]){
					upperBound[index] -= (32-shiftD[index]);
					wordCnt[index]++;
				}
				wordCnt[index] += (upperBound[index]+31) >> 5;
				numBlock[index] = (wordCnt[index]+NUMTHREAD-1)/NUMTHREAD * nStream;
				numBlock[index] = min(numBlock[index],NUMBLOCK);
			}
		}
		
		for(j=0; j<(nConInputStreams - t); j++){
			index = i;
			if (d[index] & power2[0]){
				if (shiftD[index])
					kernel3<<<numBlock[index],NUMTHREAD, 0, streams[j]>>>(gpuBitC[index]+startC[index],gpuBitD[index],shiftD[index],bitMask[32-shiftD[index]],wordCnt[index]);					
				else
					kernel4<<<numBlock[index],NUMTHREAD, 0, streams[j]>>>(gpuBitC[index]+startC[index],gpuBitD[index],wordCnt[index]);					
			}
		}
		
		for(j=0; j<(nConInputStreams - t); j++){
			index = i;
			if (lenC[index]<=(n[index]>>1)){
				cudaMemcpyAsync(gpuBitD[index],gpuBitTmp[index],sizeof(unsigned)*bitLenC[index],cudaMemcpyDeviceToDevice,streams[j]);				
				lenD[index]= lenC[index];
				lenC[index] = n[index]+1-lenC[index];
				m[index] = n[index];
			}			
			n[index]++;			
		}				
	}
	
	for(j=0; j<(nConInputStreams - t); j++){
		index = i;
		cudaMemcpyAsync(bitC[index],gpuBitC[index],sizeof(int)*bitLenS[index],cudaMemcpyDeviceToHost,streams[j]);			
	}
	
  }// end of remaining loop   
  
  for(i=0; i<nConInputStreams; i++){
	word = (lenC[i]+1) >> 5;
	bitPos = 32 - ((lenC[i]+1) & ~bitMask[27]);
	if (bitPos == 32){
		bitPos = 0;
		word--;
	}
	while(1){
		if ((bitC[i][word] & power2[bitPos]) == 0) lenC[i]--;
		else break;
		bitPos++;
		if (bitPos == 32){
		  bitPos = 0;
		  word--;
		}
	}
	if (debug){
		printf("gpuBitStream: degree is %d for input: %d\n",lenC[i],i);
		bitPrint(bitC[i],lenC[i]+1);
	}
  }
    
}

int main(int argc, char *argv[]){
  struct timeval tv1, tv2;
  int sec, usec,i;

  init(argc,argv);
  initGPU();
  printf("input length %d\n",lenS);

  gettimeofday(&tv1,NULL);
  gpuBitStream();
  gettimeofday(&tv2,NULL);
  sec = (int) (tv2.tv_sec-tv1.tv_sec);
  usec = (int) (tv2.tv_usec-tv1.tv_usec);
  if (usec < 0){
    sec--;
    usec += 1000000;
  }
  printf("gpuBitStream for %d inputs and %d streams: %f sec\n",nConInputStreams,nStream,sec+usec/1000000.0);
  
  // release resources
  for(i = 0; i < nStream; i++)
      cudaStreamDestroy(streams[i]);

  return 0;
}
