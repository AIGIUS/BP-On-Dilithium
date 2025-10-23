//
//  rng.h
//
//  Created by Bassham, Lawrence E (Fed) on 8/29/17.
//  Copyright © 2017 Bassham, Lawrence E (Fed). All rights reserved.
//
//  Modified to use standalone implementation of AES included in aes_c.c

#ifndef rng_h
#define rng_h

#include <stdio.h>
#include <stdint.h>


	void randombytes_init(unsigned int entropy_input);

	void randombytes(unsigned char* x, unsigned int xlen);

	void lfsr_update_state();


#endif /* rng_h */
