# dynamical-landscape-inference

This repository demonstrates the application of the PLNN architecture described in "[Dynamical systems theory informed learning of cellular differentiation landscapes](https://www.biorxiv.org/content/10.1101/2024.09.21.614191v1)" and implemented in a separate Github repository: [https://github.com/AddisonHowe/plnn](https://github.com/AddisonHowe/plnn).


## Setup
Basic setup, without GPU acceleration:
```bash
conda create -p ./env python=3.9 jax=0.4.23 numpy matplotlib pytorch torchvision equinox optax ipykernel pytest
conda activate env
pip install diffrax==0.4.1
```

For GPU support:
```bash
conda create -p ./env python=3.9 numpy=1.25 matplotlib=3.7 pytest=7.4 cuda-compat=12.4 tqdm ipykernel ipywidgets --yes
conda activate env
pip install --upgrade pip
pip install jax[cuda12] optax==0.1.7 diffrax==0.6.0 equinox==0.11.5 torch==2.0.1 torchvision torchaudio
```

Then, clone the PLNN project at [https://github.com/AddisonHowe/plnn](https://github.com/AddisonHowe/plnn) and install it:
```bash
git clone https://github.com/AddisonHowe/plnn.git
python -m pip install <path/to/plnn/>
```
or simply (warning, this is currently very slow) <!-- TODO -->
```bash
pip install git+https://github.com/AddisonHowe/plnn.git@dc8fb9704a65b1c84c2aa0ea715ed13d9e523956
```


## Synthetic data generation

We first generate synthetic data using the shell scripts available in `data/training_data/`.
These scripts run the PLNN module `plnn/data_generation/generate_data.py`.
Each data-generating script is described below, and will produce a subdirectory in the `data/training_data/` folder.
This subdirectory will contain three datasets: training, validation, and testing.

### Binary choice landscape

The binary choice landscape is given by 
$$\phi(x,y;\boldsymbol{\tau})=x^4+y^4+y^3-4x^2y+y^2+\tau_1x+\tau_2y.$$
We assume that two signals, $s_1$ and $s_2$, map identically to the tilt parameters, so that in terms of the signal
$$\phi(x,y;\boldsymbol{s})=x^4+y^4+y^3-4x^2y+y^2+s_1x+s_2y.$$

- [`data_phi1_1[abc]`](data/training_data/gen_data_phi1_1a.sh) \
    $T=100$, \
    $\Delta T=[10,50,100]$, \
    $\sigma=0.1$, \
    $N_{cells}=500$, \
    $N_{train}=[100,500,1000]$, \
    $N_{valid}=N_{test}=0.3N_{train}$, \
    signal switch range: $[0.1T, 0.9T]$ \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$
    
- [`data_phi1_2[abc]`](data/training_data/gen_data_phi1_2a.sh) \
    $T=20$, \
    $\Delta T=[5,10,20]$, \
    $\sigma=0.1$, \
    $N_{cells}=500$, \
    $N_{train}=[100,200,400]$, \
    $N_{valid}=N_{test}=0.2N_{train}$, \
    signal switch range: $[0.1T, 0.9T]$ \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$

- [`data_phi1_3[abc]`](data/training_data/gen_data_phi1_3a.sh) \
    $T=20$, \
    $\Delta T=[5,10,20]$, \
    $\sigma=0.1$, \
    $N_{cells}=50$, \
    $N_{train}=[50,100,200]$, \
    $N_{valid}=N_{test}=0.2N_{train}$, \
    signal switch range: $[0.1T, 0.15T]$ \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$

- [`data_phi1_4[abc]`](data/training_data/gen_data_phi1_4a.sh) \
    $T=20$, \
    $\Delta T=[5,10,20]$, \
    $\sigma=0.1$, \
    $N_{cells}=200$, \
    $N_{train}=[50,100,200]$, \
    $N_{valid}=N_{test}=0.2N_{train}$, \
    signal switch range: $[0.1T, 0.15T]$ \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$

### Binary flip landscape

The binary flip landscape is given by 
$$\phi(x,y;\boldsymbol{\tau})=x^4+y^4+x^3-2xy^2-x^2+\tau_1x+\tau_2y.$$
Again, we assume that two signals, $s_1$ and $s_2$, map identically to the tilt parameters, so that 
$$\phi(x,y;\boldsymbol{s})=x^4+y^4+x^3-2xy^2-x^2+s_1x+s_2y.$$

- [`data_phi2_1[abc]`](data/training_data/gen_data_phi2_1a.sh) \
    $T=100$, \
    $\Delta T=[10,50,100]$, \
    $\sigma=0.3$, \
    $N_{cells}=500$, \
    $N_{train}=[100,500,1000]$, \
    $N_{valid}=N_{test}=0.3N_{train}$, \
    signal switch range: $[0.1T, 0.9T]$, \
    $T_{burn}=0.05T$, \
    $s_{burn}=(-0.25, 0)$, \
    $x_0=(-1, 0)$

### Quadratic potential

The quadratic potential is given by
$$\phi(x,y;\boldsymbol{\tau})=\frac{1}{4}x^2 + \frac{1}{9}y^2 +\tau_1x+\tau_2y.$$
Here too we assume that two signals, $s_1$ and $s_2$, map identically to the tilt parameters, so that 
$$\phi(x,y;\boldsymbol{s})=\frac{1}{4}x^2 + \frac{1}{9}y^2 +s_1x+s_2y.$$

- [`data_phiq_1a`](data/training_data/gen_data_phiq_1a.sh) \
    $T=100$, \
    $\Delta T=10$, \
    $\sigma=0.5$, \
    $N_{cells}=500$, \
    $N_{train}=100$, \
    $N_{valid}=N_{test}=0.3N_{train}$, \
    signal switch range: $[0.1T, 0.9T]$, \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$

- [`data_phiq_2a`](data/training_data/gen_data_phiq_2a.sh) \
    $T=100$, \
    $\Delta T=10$, \
    $\sigma=0.1$, \
    $N_{cells}=500$, \
    $N_{train}=100$, \
    $N_{valid}=N_{test}=0.3N_{train}$, \
    signal switch range: $[0.1T, 0.9T]$, \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$


## PLNN model training and evaluation

We apply the PLNN model training procedure to the synthetic datasets.
The directories within `data/model_training_args` contain a number of tsv files, each specifying arguments to be used for an instance of model training.
The list of these tsv files is included below, and a table summarizing the various arguments contained in each can be found at [data/model_training_args/README.md](data/model_training_args/README.md).
(This table can be updated to reflect the argument files using the command `sh data/model_training_args/_update_readme.sh`)

The shell script `scripting/echo_training_cmd_from_runargs.sh` takes as input a path specifying one of these argument files, and prints the corresponding python command that will run the training procedure.
**Note**: The argument files contained here assume access to a GPU.

For each argument file, we can train multiple models in order to assess the variation across multiple training runs.
A full list of the trained models can be found at [data/trained_models/README.md](data/trained_models/README.md).
The training process generates an output model directory containing log files and data saved over the course of training.


### Model training argument files
#### Binary choice landscape

- `data_phi1_1[abc]`
    - [run_phi1_1a_v_mmd1](data/model_training_args/synbindec/run_phi1_1a_v_mmd1.tsv)
    - [run_phi1_1b_v_mmd1](data/model_training_args/synbindec/run_phi1_1b_v_mmd1.tsv) 
    - [run_phi1_1c_v_mmd1](data/model_training_args/synbindec/run_phi1_1c_v_mmd1.tsv)
    
- `data_phi1_2[abc]`
    - [run_phi1_2a_v_mmd1](data/model_training_args/synbindec/run_phi1_2a_v_mmd1.tsv)   
    - [run_phi1_2b_v_mmd1](data/model_training_args/synbindec/run_phi1_2b_v_mmd1.tsv)    
    - [run_phi1_2c_v_mmd1](data/model_training_args/synbindec/run_phi1_2c_v_mmd1.tsv)   

- `data_phi1_3[abc]`
    - [run_phi1_3a_v_mmd1](data/model_training_args/synbindec/run_phi1_3a_v_mmd1.tsv)
    - [run_phi1_3b_v_mmd1](data/model_training_args/synbindec/run_phi1_3b_v_mmd1.tsv)
    - [run_phi1_3c_v_mmd1](data/model_training_args/synbindec/run_phi1_3c_v_mmd1.tsv)

- `data_phi1_4[abc]`
    - [run_phi1_4a_v_mmd1](data/model_training_args/synbindec/run_phi1_4a_v_mmd1.tsv)
    - [run_phi1_4b_v_mmd1](data/model_training_args/synbindec/run_phi1_4b_v_mmd1.tsv)
    - [run_phi1_4c_v_mmd1](data/model_training_args/synbindec/run_phi1_4c_v_mmd1.tsv)

#### Binary flip landscape

- `data_phi2_1[abc]`
    - [run_phi2_1a_v_mmd1](data/model_training_args/synbindec/run_phi2_1a_v_mmd1.tsv)   
    - [run_phi2_1b_v_mmd1](data/model_training_args/synbindec/run_phi2_1b_v_mmd1.tsv)   
    - [run_phi2_1c_v_mmd1](data/model_training_args/synbindec/run_phi2_1c_v_mmd1.tsv)   

#### Quadratic potential

- `data_phiq_1a`
    - [run_phiq_1a_v_mmd1](data/model_training_args/quadratic/run_phiq_1a_v_mmd1.tsv)   
- `data_phiq_2a`
    - [run_phiq_2a_v_mmd1](data/model_training_args/quadratic/run_phiq_2a_v_mmd1.tsv)   


### Model evaluation
After the training process, we evaluate the resulting model on data not seen during the training process.


# Acknowledgments
This work was inspired by the work of Sáez et al. in [Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions](https://pubmed.ncbi.nlm.nih.gov/34536382/).


# References

[1] Sáez M, Blassberg R, Camacho-Aguilar E, Siggia ED, Rand DA, Briscoe J. Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions. Cell Syst. 2022 Jan 19;13(1):12-28.e3. doi: 10.1016/j.cels.2021.08.013. Epub 2021 Sep 17. PMID: 34536382; PMCID: PMC8785827.
