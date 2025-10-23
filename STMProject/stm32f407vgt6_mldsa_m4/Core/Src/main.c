/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "rng.h"
#include "api.h"
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
size_t mlen=0, smlen=0, mlen2=0;
uint8_t m[1024], m2[1024];
uint8_t sm[1024+CRYPTO_BYTES];
uint8_t pk[CRYPTO_PUBLICKEYBYTES];
uint8_t sk[CRYPTO_SECRETKEYBYTES];
unsigned int seed = 0x12345678;
char done_flag[5] = "done\n";
int rej_num = 0;
int trigger_idx = 0;
int poly_idx = 0;
unsigned char tmp[4];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void cal_sign_rej_num(){
	randombytes_init(seed);
	for(int i=0;i<100;++i)
		randombytes(tmp,4);
	rej_num = 0;
	mlen = 0;
	randombytes(&mlen,4);
	mlen &= 0xff;
	randombytes(m,mlen);
	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_0, GPIO_PIN_RESET);
	crypto_sign(sm, &smlen, m, mlen, sk, -2, -2, &rej_num);
	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_0, GPIO_PIN_SET);
}

void sign(){
	randombytes_init(seed);
	for(int i=0;i<100;++i)
		randombytes(tmp,4);
	mlen = 0;
	randombytes(&mlen,4);
	mlen &= 0xff;
	randombytes(m,mlen);
	crypto_sign(sm, &smlen, m, mlen, sk, trigger_idx, poly_idx, &rej_num);
}

void test_algorithm(){
	int ret = 0;

	ret = crypto_sign_keypair(pk, sk);
	if(ret!=0)
		HAL_UART_Transmit(&huart2,"key gen wrong\n",strlen("key gen wrong\n"),1000);
	mlen = 33;
	randombytes(m,mlen);
	ret = crypto_sign(sm, &smlen, m, mlen, sk,-2,-2,&rej_num);
//	ret = crypto_sign(sm, &smlen, m, mlen, sk);
	if(ret!=0)
		HAL_UART_Transmit(&huart2,"sign wrong\n",strlen("sign wrong\n"),1000);
	ret = crypto_sign_open(m2,&mlen2,sm,smlen,pk);
	if(ret!=0)
		HAL_UART_Transmit(&huart2,"sign open wrong\n",strlen("sign open wrong\n"),1000);
	if(mlen != mlen2)
		HAL_UART_Transmit(&huart2,"mlen2 wrong\n",strlen("mlen2 wrong\n"),1000);
	for(int i=0;i<mlen;++i){
		if(m[i]!=m2[i]){
			HAL_UART_Transmit(&huart2,"m2 wrong\n",strlen("m2 wrong\n"),1000);
			return;
		}
	}
	HAL_UART_Transmit(&huart2,"sign test success\n",strlen("sign test success\n"),1000);
}
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
  HAL_GPIO_WritePin(GPIOC, GPIO_PIN_0, GPIO_PIN_SET);
  	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_1, GPIO_PIN_SET);
  	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_2, GPIO_PIN_SET);
  	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_3, GPIO_PIN_SET);
  	unsigned char op = 0;
  	randombytes_init(0x3f3f3f3f);
  	for(int i=0;i<100;++i)
  		randombytes(tmp,4);
  	crypto_sign_keypair(pk, sk); //fix key by fix seed
      __HAL_FLASH_PREFETCH_BUFFER_DISABLE();
      __HAL_FLASH_DATA_CACHE_DISABLE();
      __disable_irq();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  op=0;
	  HAL_UART_Receive(&huart2,&op,1,1000);
	  switch(op){
	  case 0x01: // get seed and return rej num
		  HAL_UART_Receive(&huart2,(unsigned char*)&seed,sizeof(unsigned int),1000);
		  cal_sign_rej_num();
		  HAL_UART_Transmit(&huart2,(unsigned char*)&rej_num,4,1000);
		  break;
	  case 0x02: //sign message
		  HAL_UART_Receive(&huart2,(unsigned char*)&seed,sizeof(unsigned int),1000);
		  HAL_UART_Receive(&huart2,(unsigned char*)&trigger_idx,sizeof(int),1000);
		  HAL_UART_Receive(&huart2,(unsigned char*)&poly_idx,sizeof(int),1000);
		  sign();
		  HAL_UART_Transmit(&huart2,(unsigned char*)done_flag,strlen(done_flag),1000);
		  break;
	  case 0x03: // gen key
			randombytes_init(0x3f3f3f3f);
			for(int i=0;i<100;++i)
				randombytes(tmp,4);
			crypto_sign_keypair(pk, sk); //fix key by fix seed
	  case 0x05:
		  test_algorithm();
	  default:
		  op = 0;
		  trigger_idx = 0;
	  }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
