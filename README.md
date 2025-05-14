# masters-thesis-VLN

Model training and fine-tuning were conducted using Jupyter Notebooks. The fine-tuning code for both panoramic and low-level action spaces is available in the notebook files.

The `scripts/` folder contains the preprocessing scripts used to prepare the R2R dataset for training and evaluation. This includes formatting the data to work with both the panoramic and low-level action spaces.

To use this codebase, you must place this repository inside the `Matterport3D` simulator directory. The simulator can be found here:  
[https://github.com/peteanderson80/Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator)

The `requirements.txt` file contains the Python dependencies used for training the models (the rest is automatically installed with the matterport3D docker image).  To install the required dependencies, activate your virtual environment and run:

```bash
pip install -r requirements.txt
```
