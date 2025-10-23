//
//  rng.c
//
//  Created by Bassham, Lawrence E (Fed) on 8/29/17.
//  Copyright © 2017 Bassham, Lawrence E (Fed). All rights reserved.
//
//  Modified to use standalone implementation of AES included in aes_c.c

#include <string.h>
#include "rng.h"

static unsigned int lfsr_state;
static unsigned int g1, g2, g22, g32;

void lfsr_update_state() {
    g1 = lfsr_state;
    g2 = lfsr_state >> 1;
    g22 = lfsr_state >> 21;
    g32 = lfsr_state >> 31;
    lfsr_state <<= 1;
    lfsr_state ^= ((g1 ^ g2 ^ g22 ^ g32) & 0x1);
}

void
randombytes_init(unsigned int entropy_input)
{
    //memcpy(&lfsr_state, entropy_input, 4);
    lfsr_state = entropy_input;
}

void
randombytes(unsigned char* x, unsigned int xlen)
{

    for (int i = 0; i < xlen; ++i) {
        lfsr_update_state();
        x[i] = (unsigned char)lfsr_state;
    }
}
