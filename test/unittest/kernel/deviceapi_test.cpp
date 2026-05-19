#include "deviceapi_test.hpp"

// Static member definitions
flagcxHandlerGroup_t DeviceApiTest::handler = nullptr;
flagcxComm_t DeviceApiTest::comm = nullptr;
flagcxStream_t DeviceApiTest::stream = nullptr;
void *DeviceApiTest::devBuff = nullptr;
size_t DeviceApiTest::size = 0;

void DeviceApiTest::SetUpTestSuite() {
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size = DEVICEAPI_TEST_SIZE;

  flagcxHandleInit(&handler);
  flagcxDeviceHandle_t &devHandle = handler->devHandle;

  int numDevices;
  devHandle->getDeviceCount(&numDevices);
  devHandle->setDevice(rank % numDevices);

  flagcxUniqueId_t uniqueId = nullptr;
  if (rank == 0)
    flagcxGetUniqueId(&uniqueId);
  else
    uniqueId = (flagcxUniqueId_t)calloc(1, sizeof(flagcxUniqueId));
  MPI_Bcast((void *)uniqueId, sizeof(flagcxUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommInitRank(&comm, nranks, uniqueId, rank);
  free(uniqueId);

  devHandle->streamCreate(&stream);

  flagcxMemAlloc(&devBuff, size);
}

void DeviceApiTest::TearDownTestSuite() {
  if (handler == nullptr)
    return;

  handler->devHandle->streamDestroy(stream);

  if (comm)
    flagcxCommDestroy(comm);

  flagcxMemFree(devBuff);

  flagcxHandleFree(handler);

  handler = nullptr;
  comm = nullptr;
  stream = nullptr;
  devBuff = nullptr;
}

void DeviceApiTest::SetUp() {
  FlagCXTest::SetUp();
  if (comm == nullptr) {
    GTEST_SKIP() << "SetUpTestSuite failed";
  }
}
