# masters-thesis-VLN

This repository contains the codebase used for the experiments in:

> **[Following Route Instructions using Large Vision-Language Models: A Comparison between Low-level and Panoramic Action Spaces](https://aclanthology.org/2025.icnlsp-1.43/)**  
> Vebjørn Kåsene, Pierre Lison (ICNLSP 2025)

Model training and fine-tuning were conducted using Jupyter Notebooks. The fine-tuning code for both panoramic and low-level action spaces is available in the notebook files.

The `scripts/` folder contains the preprocessing scripts used to prepare the R2R dataset for training and evaluation. This includes formatting the data for both panoramic and low-level action spaces.

To use this codebase, you must place this repository inside the `Matterport3D` simulator directory. The simulator can be found here:  
[https://github.com/peteanderson80/Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator)

The `requirements.txt` file contains the Python dependencies used for training the models (the rest is automatically installed with the Matterport3D Docker image). To install the required dependencies, activate your virtual environment and run:

```bash
pip install -r requirements.txt
```

## Fine-tuned Model Weights

We provide the fine-tuned model checkpoints used in the paper:

- **Panoramic action space model (Qwen2.5-VL)**  
  https://huggingface.co/Vebbern/Qwen2.5-VL-3B-R2R-panoramic

- **Low-level action space model (Qwen2.5-VL)**  
  https://huggingface.co/Vebbern/Qwen2.5-VL-3B-R2R-low-level

## Reference

If you use this codebase in your research, please cite the following paper:

Kåsene, V., & Lison, P. (2025).  
*Following Route Instructions using Large Vision-Language Models: A Comparison between Low-level and Panoramic Action Spaces.*  
Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP-2025), 449–463.  
https://aclanthology.org/2025.icnlsp-1.43/

```bibtex
@inproceedings{kasene-lison-2025-following,
    title = "Following Route Instructions using Large Vision-Language Models: A Comparison between Low-level and Panoramic Action Spaces",
    author = "K{\r{a}}sene, Vebj{\o}rn  and
      Lison, Pierre",
    editor = "Abbas, Mourad  and
      Yousef, Tariq  and
      Galke, Lukas",
    booktitle = "Proceedings of the 8th International Conference on Natural Language and Speech Processing (ICNLSP-2025)",
    month = aug,
    year = "2025",
    address = "Southern Denmark University, Odense, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.icnlsp-1.43/",
    pages = "449--463"
}
```
