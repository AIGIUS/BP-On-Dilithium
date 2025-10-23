################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/fips202.c \
../Core/Src/gpio.c \
../Core/Src/keccakf1600.c \
../Core/Src/main.c \
../Core/Src/ntt.c \
../Core/Src/packing.c \
../Core/Src/poly.c \
../Core/Src/polyvec.c \
../Core/Src/reduce.c \
../Core/Src/rng.c \
../Core/Src/rounding.c \
../Core/Src/sign.c \
../Core/Src/stm32f4xx_hal_msp.c \
../Core/Src/stm32f4xx_it.c \
../Core/Src/symmetric-shake.c \
../Core/Src/syscalls.c \
../Core/Src/sysmem.c \
../Core/Src/system_stm32f4xx.c \
../Core/Src/usart.c 

OBJS += \
./Core/Src/fips202.o \
./Core/Src/gpio.o \
./Core/Src/keccakf1600.o \
./Core/Src/main.o \
./Core/Src/ntt.o \
./Core/Src/packing.o \
./Core/Src/poly.o \
./Core/Src/polyvec.o \
./Core/Src/reduce.o \
./Core/Src/rng.o \
./Core/Src/rounding.o \
./Core/Src/sign.o \
./Core/Src/stm32f4xx_hal_msp.o \
./Core/Src/stm32f4xx_it.o \
./Core/Src/symmetric-shake.o \
./Core/Src/syscalls.o \
./Core/Src/sysmem.o \
./Core/Src/system_stm32f4xx.o \
./Core/Src/usart.o 

C_DEPS += \
./Core/Src/fips202.d \
./Core/Src/gpio.d \
./Core/Src/keccakf1600.d \
./Core/Src/main.d \
./Core/Src/ntt.d \
./Core/Src/packing.d \
./Core/Src/poly.d \
./Core/Src/polyvec.d \
./Core/Src/reduce.d \
./Core/Src/rng.d \
./Core/Src/rounding.d \
./Core/Src/sign.d \
./Core/Src/stm32f4xx_hal_msp.d \
./Core/Src/stm32f4xx_it.d \
./Core/Src/symmetric-shake.d \
./Core/Src/syscalls.d \
./Core/Src/sysmem.d \
./Core/Src/system_stm32f4xx.d \
./Core/Src/usart.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/%.o Core/Src/%.su Core/Src/%.cyclo: ../Core/Src/%.c Core/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F407xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O3 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src

clean-Core-2f-Src:
	-$(RM) ./Core/Src/fips202.cyclo ./Core/Src/fips202.d ./Core/Src/fips202.o ./Core/Src/fips202.su ./Core/Src/gpio.cyclo ./Core/Src/gpio.d ./Core/Src/gpio.o ./Core/Src/gpio.su ./Core/Src/keccakf1600.cyclo ./Core/Src/keccakf1600.d ./Core/Src/keccakf1600.o ./Core/Src/keccakf1600.su ./Core/Src/main.cyclo ./Core/Src/main.d ./Core/Src/main.o ./Core/Src/main.su ./Core/Src/ntt.cyclo ./Core/Src/ntt.d ./Core/Src/ntt.o ./Core/Src/ntt.su ./Core/Src/packing.cyclo ./Core/Src/packing.d ./Core/Src/packing.o ./Core/Src/packing.su ./Core/Src/poly.cyclo ./Core/Src/poly.d ./Core/Src/poly.o ./Core/Src/poly.su ./Core/Src/polyvec.cyclo ./Core/Src/polyvec.d ./Core/Src/polyvec.o ./Core/Src/polyvec.su ./Core/Src/reduce.cyclo ./Core/Src/reduce.d ./Core/Src/reduce.o ./Core/Src/reduce.su ./Core/Src/rng.cyclo ./Core/Src/rng.d ./Core/Src/rng.o ./Core/Src/rng.su ./Core/Src/rounding.cyclo ./Core/Src/rounding.d ./Core/Src/rounding.o ./Core/Src/rounding.su ./Core/Src/sign.cyclo ./Core/Src/sign.d ./Core/Src/sign.o ./Core/Src/sign.su ./Core/Src/stm32f4xx_hal_msp.cyclo ./Core/Src/stm32f4xx_hal_msp.d ./Core/Src/stm32f4xx_hal_msp.o ./Core/Src/stm32f4xx_hal_msp.su ./Core/Src/stm32f4xx_it.cyclo ./Core/Src/stm32f4xx_it.d ./Core/Src/stm32f4xx_it.o ./Core/Src/stm32f4xx_it.su ./Core/Src/symmetric-shake.cyclo ./Core/Src/symmetric-shake.d ./Core/Src/symmetric-shake.o ./Core/Src/symmetric-shake.su ./Core/Src/syscalls.cyclo ./Core/Src/syscalls.d ./Core/Src/syscalls.o ./Core/Src/syscalls.su ./Core/Src/sysmem.cyclo ./Core/Src/sysmem.d ./Core/Src/sysmem.o ./Core/Src/sysmem.su ./Core/Src/system_stm32f4xx.cyclo ./Core/Src/system_stm32f4xx.d ./Core/Src/system_stm32f4xx.o ./Core/Src/system_stm32f4xx.su ./Core/Src/usart.cyclo ./Core/Src/usart.d ./Core/Src/usart.o ./Core/Src/usart.su

.PHONY: clean-Core-2f-Src

