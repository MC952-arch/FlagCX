#pragma once

#include "flagcx.h"
#include "flagcx_test.hpp"

class FlagCXTopoTest : public FlagCXTest {
protected:
  void SetUp() override;
  void TearDown() override;

  flagcxDeviceHandle_t devHandle = nullptr;
  flagcxComm_t comm = nullptr;
  flagcxUniqueId uniqueId;
};
