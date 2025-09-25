# Thesis Code Repository

This repository contains selected code used for my Master's thesis:  
**"Explainability-Driven Evaluation of Speaker Anonymisation"** (TU Berlin, 2025).

**Important note:**  
This is **research code** and was written primarily for my own experiments.  
It is **not optimised for reuse, not guaranteed to run out-of-the-box, and has not been systematically tested**.  
The repository serves only to **illustrate the main scripts and workflows** behind the experiments described in my thesis.

---

## Disclaimer

- All code in this repository was written by me, unless otherwise noted in comments or attributions.
- I occasionally used **ChatGPT (OpenAI)** and **Claude (Anthropic)** as coding assistants (e.g., for debugging, clean code, code structure or boilerplate generation).  
  Final implementations, integration, and research design remain my own responsibility.
- No guarantees are provided that this code will run without modification. Use at your own risk.

---

## Repository Structure

- `anon/` - scripts for anonymising datasets with SA-Toolkit, kNN-VC and Private-kNN-VC.
- `data/` - scripts for downloading and pre-processing the data
- `embeddings/` - scripts for optaining embeddings and needed pkls from SCOTUS- and VoxCeleb-datasets
- `training/` - scripts to train attribute classifiers, evaluate them, train Random Forest for speaker verification, and evaluate them.

_Note:_ File organisation reflects my working setup and may not be fully generalised.

---

## Usage

This repository is **not intended as a plug-and-play package**.  
Running the code requires:

- Access to the datasets described in the thesis (e.g., VoxCeleb2, SCOTUS).
- Correct environment setup (Python version, dependencies, and hardware requirements).
- Adjustments to paths and configuration parameters.

Due to dataset licensing, **data and pretrained models are not included**.

---

## License

This repository is provided _as is_ for academic transparency.  
No warranty, support, or guarantee of correctness is provided.  
If you reuse parts of the code, please cite my thesis.

---

## Contact

If you have questions about the research, please contact me at:  
**[aljoscha.lipski@googlemail.com]**

---

## Citation

If you use insights or scripts from this repo, please cite my thesis.
