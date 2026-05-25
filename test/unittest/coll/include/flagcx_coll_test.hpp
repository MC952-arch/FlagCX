#pragma once

#include "flagcx.h"
#include "flagcx_test.hpp"

class FlagCXCollTest : public FlagCXTest {
protected:
  FlagCXCollTest() {}

  void SetUp();

  void TearDown();

  void Run();

  flagcxDeviceHandle_t devHandle;
  flagcxComm_t comm;
  flagcxStream_t stream;
  void *sendbuff;
  void *recvbuff;
  void *hostsendbuff;
  void *hostrecvbuff;
  size_t size;
  size_t count;
};
