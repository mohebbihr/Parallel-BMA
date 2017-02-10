#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define min(a,b) ((a) < (b) ? (a) : (b))
#define bufferSize 1024
#define NUMTHREAD 256
#define NUMBLOCK 64

char buffer[bufferSize];
int deviceNum = 1, debug = 0, randGen = 0, lenS, m;
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

  /* add nother input named (nSerialInputs) that determines the number of input streams */  
  if (argc < 3){
    printf("Usage: ./gpuBitStream filename length -s NumberofSerialInputs -b debugLvl -d deviceNum\n");
    printf("\tto generate random string: ./gpuBitStream randGen length\n");
    exit(1);
  }
  if (argc > 3){
    i = 3;
    while (i<argc){
      if (!strcmp(argv[i],"-s")) sscanf(argv[i+1],"%d",&nSerialInputs);
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
  
  /* we have to have nSerialInputs array's for n serial inputs*/
  bitRS = new unsigned**[nSerialInputs];
  bitC = new unsigned*[nSerialInputs];
  bitD = new unsigned*[nSerialInputs];
  bitTmp = new unsigned*[nSerialInputs];
  bitLenS = new unsigned[nSerialInputs];
  bitLenC = new unsigned[nSerialInputs];
  
  for(i=0; i<nSerialInputs; i++){
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
  
  /* we have to change the memory allocation for gpu as well as we done for cpu */
  gpuBitRS = new unsigned**[nSerialInputs];
  gpuBitC = new unsigned*[nSerialInputs];
  gpuBitD = new unsigned*[nSerialInputs];
  gpuBitTmp = new unsigned*[nSerialInputs];
  gpuParity = new unsigned*[nSerialInputs];
  
  // allocate and initialize an array of stream handles
  streams = (cudaStream_t*) malloc(1 * sizeof(cudaStream_t));
  for(i=0; i<1; i++)
	cudaStreamCreate(&(streams[i]));
  
  for(j=0; j<nSerialInputs; j++){	
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

void gpuBitSerial(int input_index){
  // the input_index determines which serial input should be executed
  int i, numBlock, q, r, upperBound, wordCnt, shiftD, startC, word, bitPos;
  unsigned d;

  bitD[input_index][0] = bitC[input_index][0] = power2[31];
  for(i=1;i<bitLenS[input_index];i++) bitD[input_index][i] = bitC[input_index][i] = 0;
  cudaMemcpyAsync(gpuBitC[input_index],bitC[input_index],sizeof(int)*bitLenS[input_index],cudaMemcpyHostToDevice,streams[0]);
  cudaMemcpyAsync(gpuBitD[input_index],bitD[input_index]-1,sizeof(int)*(bitLenS[input_index]+1),cudaMemcpyHostToDevice,streams[0]);
  gpuBitD[input_index]++;
  n = lenC = lenD = 0;
  m = -1;
  while (n<lenS){
    q = (lenS-1-n) >> 5;
    r = (lenS-1-n) & ~bitMask[27];
    bitLenC[input_index] = (lenC+1+31)>>5;
    numBlock = (bitLenC[input_index]+NUMTHREAD-1)/NUMTHREAD;
    numBlock = min(numBlock,NUMBLOCK);
    kernel1<<<numBlock,NUMTHREAD, 0, streams[0]>>>(gpuBitRS[input_index][r]+q,gpuBitC[input_index],gpuParity[input_index],bitLenC[input_index]);    
    kernel2<<<1,NUMBLOCK, 0, streams[0]>>>(gpuParity[input_index],numBlock);    
    cudaMemcpyAsync(&d,gpuParity[input_index],sizeof(unsigned),cudaMemcpyDeviceToHost,streams[0]);
    d = d - ((d >> 1) & 0x55555555);
    d = (d & 0x33333333) + ((d >> 2) & 0x33333333);
    d = (((d + (d >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    if (d & power2[0]){
      if (lenC<=(n>>1))
		cudaMemcpyAsync(gpuBitTmp[input_index],gpuBitC[input_index],sizeof(unsigned)*bitLenC[input_index],cudaMemcpyDeviceToDevice,streams[0]);
      upperBound = min(lenD+1,lenS+m-n);
      startC = (n-m) >> 5;
      shiftD = (n-m) & ~bitMask[27];
      wordCnt = 0;
      if (shiftD){
		upperBound -= (32-shiftD);
		wordCnt++;
      }
      wordCnt += (upperBound+31) >> 5;
      numBlock = (wordCnt+NUMTHREAD-1)/NUMTHREAD;
      numBlock = min(numBlock,NUMBLOCK);
      if (shiftD)
		kernel3<<<numBlock,NUMTHREAD, 0, streams[0]>>>(gpuBitC[input_index]+startC,gpuBitD[input_index],shiftD,bitMask[32-shiftD],wordCnt);
      else
		kernel4<<<numBlock,NUMTHREAD, 0, streams[0]>>>(gpuBitC[input_index]+startC,gpuBitD[input_index],wordCnt);      
      if (lenC<=(n>>1)){
		cudaMemcpyAsync(gpuBitD[input_index],gpuBitTmp[input_index],sizeof(unsigned)*bitLenC[input_index],cudaMemcpyDeviceToDevice,streams[0]);
		lenD = lenC;
		lenC = n+1-lenC;
		m = n;
      }
    }
    n++;
  }
  cudaMemcpyAsync(bitC[input_index],gpuBitC[input_index],sizeof(int)*bitLenS[input_index],cudaMemcpyDeviceToHost,streams[0]);
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
    printf("gpuBitSerial: degree is %d for input: %d\n",lenC,input_index);
    bitPrint(bitC[input_index],lenC+1);
  }
}

int main(int argc, char *argv[]){
  struct timeval tv1, tv2;
  int sec, usec,i;

  init(argc,argv);
  initGPU();
  printf("input length %d\n",lenS);

  gettimeofday(&tv1,NULL);
  for(i=0; i<nSerialInputs; i++)
	gpuBitSerial(i);
  gettimeofday(&tv2,NULL);
  sec = (int) (tv2.tv_sec-tv1.tv_sec);
  usec = (int) (tv2.tv_usec-tv1.tv_usec);
  if (usec < 0){
    sec--;
    usec += 1000000;
  }
  printf("gpuBitSerial for %d inputs: %f sec\n",nSerialInputs,sec+usec/1000000.0);

  return 0;
}
