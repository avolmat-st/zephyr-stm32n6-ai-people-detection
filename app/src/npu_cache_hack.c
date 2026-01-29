/**
 ******************************************************************************
 * @file    npu_cache_hack.c
 * @author  GPM Application Team
 *
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#include "stm32n6xx_hal.h"

/* 3.0.0 version of ../../Lib/AI_Runtime/Npu/Devices/STM32N6xx/npu_cache.c will not compile in zephyr due bad zephyr
 * hal order.
 * This file add stm32n6xx_hal.h so we include hal tpes before compiling original npu_cache.c.
 */

#include "../../Lib/AI_Runtime/Npu/Devices/STM32N6xx/npu_cache.c"
