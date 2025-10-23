# Belief-Propagation-On-Dilithium

## Project Overview

This repository contains the experimental code and data for the paper **"Release the Power of Rejected Signatures: An Efficient Side-Channel Attack on Dilithium"**.

**Paper Link**: [https://eprint.iacr.org/2025/582](https://eprint.iacr.org/2025/582)

## Abstract

The Module-Lattice-Based Digital Signature Standard (ML-DSA), formerly known as CRYSTALS-Dilithium, is a lattice-based post-quantum cryptographic scheme. In August 2024, the National Institute of Standards and Technology (NIST) officially standardized ML-DSA under FIPS 204. Dilithium generates one valid signature and multiple rejected signatures during the signing process. Most Side-Channel Attacks targeting Dilithium have focused solely on the valid signature, while neglecting the hints contained in rejected signatures.

This paper proposes a method for recovering the private key by simultaneously leveraging side-channel leakages from both valid signatures and rejected signatures. This approach minimizes the number of signing attempts required for full key recovery. We construct a factor graph incorporating all relevant side-channel leakages and apply the Belief Propagation (BP) algorithm for private key recovery.

Our proof-of-concept experiment conducted on a Cortex M4 core chip demonstrates that utilizing rejected signatures reduces the required number of traces by at least 50% for full key recovery. 

## Project Structure

```
BP-On-Dilithium/
├── STMProject/          # STM32 Project Files
│   ├── stm32f407vgt6_mldsa_ref/    # ML-DSA-44 Reference Implementation
│   └── stm32f407vgt6_mldsa_m4/     # ML-DSA-44 M4 Optimized Implementation
├── EXP/                 # Experimental Directory
│   ├── data/            # Algorithm Intermediate Values
│   ├── exp_res/         # Experimental Results
│   ├── py_code/         # Python Scripts
│   └── traces/          # Collected Waveform Data
└── README.md            # Project Documentation
```

### Directory Description

#### STMProject/
Contains two STM32CubeIDE projects:
- **stm32f407vgt6_mldsa_ref**: ML-DSA-44 reference implementation
- **stm32f407vgt6_mldsa_m4**:  ML-DSA-44 ASM-optimized implementation

Interested readers can flash these projects to STM32 development boards to reproduce the traces collection experiments.

#### EXP/
Experimental data and code directory:
- **data/**: Stores intermediate values during algorithm execution
- **exp_res/**: Stores experimental results and analysis reports
- **py_code/**: Contains Belief Propagation algorithm implementation and data analysis scripts
- **traces/**: Stores side-channel traces collected from STM32 devices

## Important Notes

⚠️ **Version Information**: The uploaded data corresponds to the **second version** of the paper on ePrint, not the first version. The second version includes additional experimental data and result analysis compared to the first version.

## Citation

If you use the code or data from this project, please cite the following paper:

```bibtex
@misc{cryptoeprint:2025/582,
      author = {Zheng Liu and An Wang and Congming Wei and Yaoling Ding and Jingqi Zhang and Annyu Liu and Liehuang Zhu},
      title = {Release the Power of Rejected Signatures: An Efficient Side-Channel Attack on Dilithium},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/582},
      year = {2025},
      url = {https://eprint.iacr.org/2025/582}
}
```

## Contact

For questions or suggestions, please contact us through:
- Author Email: lzz73092@gmail.com
- Project Issues: Please submit issues on GitHub

---

**Disclaimer**: This project is for academic research purposes only. Please do not use the related techniques for malicious attacks or illegal purposes.