# masters-thesis-VLN

Model training and fine-tuning were conducted using Jupyter Notebooks. The fine-tuning code for both panoramic and low-level action spaces is available in the notebook files.

The `scripts/` folder contains the preprocessing scripts used to prepare the R2R dataset for training and evaluation. This includes formatting the data to work with both the panoramic and low-level action spaces.

To use this codebase, you must place this repository inside the `Matterport3D` simulator directory. The simulator can be found here:  
[https://github.com/peteanderson80/Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator)

The `requirements.txt` file contains the Python dependencies used for training the models (the rest is automatically installed with the matterport3D docker image).  To install the required dependencies, activate your virtual environment and run:

```bash
pip install -r requirements.txt
```

### API Use for Remote Evaluation

In cases where the Matterport3D simulator cannot be run on the machine used for training (e.g., due to system incompatibility, limited GPU access, or resource constraints), this codebase supports evaluation via a remote API.

The API allows a separate machine — where the Matterport3D simulator is running — to handle simulation and evaluation requests. 
