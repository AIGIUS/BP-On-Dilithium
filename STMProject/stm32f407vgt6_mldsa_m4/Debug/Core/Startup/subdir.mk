################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../Core/Startup/ntt.s \
../Core/Startup/pointwise_mont.s \
../Core/Startup/startup_stm32f407vgtx.s \
../Core/Startup/vector.s 

OBJS += \
./Core/Startup/ntt.o \
./Core/Startup/pointwise_mont.o \
./Core/Startup/startup_stm32f407vgtx.o \
./Core/Startup/vector.o 

S_DEPS += \
./Core/Startup/ntt.d \
./Core/Startup/pointwise_mont.d \
./Core/Startup/startup_stm32f407vgtx.d \
./Core/Startup/vector.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Startup/%.o: ../Core/Startup/%.s Core/Startup/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m4 -g3 -DDEBUG -c -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

clean: clean-Core-2f-Startup

clean-Core-2f-Startup:
	-$(RM) ./Core/Startup/ntt.d ./Core/Startup/ntt.o ./Core/Startup/pointwise_mont.d ./Core/Startup/pointwise_mont.o ./Core/Startup/startup_stm32f407vgtx.d ./Core/Startup/startup_stm32f407vgtx.o ./Core/Startup/vector.d ./Core/Startup/vector.o

.PHONY: clean-Core-2f-Startup

