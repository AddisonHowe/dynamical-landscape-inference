# dynamical-landscape-inference

This repository demonstrates the application of the PLNN architecture described in [Dynamical systems theory informed learning of cellular differentiation landscapes](https://www.biorxiv.org/content/10.1101/2024.09.21.614191v1) and implemented in a separate Github repository: [https://github.com/AddisonHowe/plnn](https://github.com/AddisonHowe/plnn).


## Setup
Basic setup, without GPU capabilities:
```bash
mamba create -p ./env python=3.9 jax=0.4.23 numpy matplotlib pytorch torchvision equinox optax ipykernel pytest
mamba activate env
pip install diffrax==0.4.1
```

For GPU support, specifying cuda toolkit 11.2:
```bash
mamba create -p <env-path> python=3.9 pytorch=1.11[build=cuda112*] numpy=1.25 matplotlib=3.7 pytest=7.4 tqdm ipykernel ipywidgets
mamba activate env
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optax==0.1.7 diffrax==0.4.1 equinox==0.11.2
```

Then, clone the PLNN project at [https://github.com/AddisonHowe/plnn](https://github.com/AddisonHowe/plnn) and install it:
```bash
git clone https://github.com/AddisonHowe/plnn.git
python -m pip install <path/to/plnn/>
```


## Synthetic data generation

We first generate synthetic data using the shell scripts available in `data/training_data/`.
These scripts run the PLNN module `plnn/data_generation/generate_data.py`.
Each data-generating script is described below, and will produce a subdirectory in the `data/training_data/` folder.
This subdirectory will contain three datasets: training, validation, and testing.

### Binary choice landscape

- [`data_phi1_1[abc]`](data/training_data/gen_data_phi1_1a.sh) \
    $t\in[0,100]$, \
    $\Delta T=[10,50,100]$, \
    $\sigma=0.1$, \
    $N_{cells}=500$, \
    $N_{train}=[100,500,1000]$, \
    $N_{valid}=N_{test}=0.3N_{train}$, \
    signal switch range: $[0.1t_{fin}, 0.9t_{fin}]$ \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$
    
- [`data_phi1_2[abc]`](data/training_data/gen_data_phi1_2a.sh) \
    $t\in[0,20]$, \
    $\Delta T=[5,10,20]$, \
    $\sigma=0.1$, \
    $N_{cells}=500$, \
    $N_{train}=[100,200,400]$, \
    $N_{valid}=N_{test}=0.2N_{train}$, \
    signal switch range: $[0.1t_{fin}, 0.9t_{fin}]$ \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$

- [`data_phi1_3[abc]`](data/training_data/gen_data_phi1_3a.sh) \
    $t\in[0,20]$, \
    $\Delta T=[5,10,20]$, \
    $\sigma=0.1$, \
    $N_{cells}=50$, \
    $N_{train}=[50,100,200]$, \
    $N_{valid}=N_{test}=0.2N_{train}$, \
    signal switch range: $[0.1t_{fin}, 0.15t_{fin}]$ \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$

- [`data_phi1_4[abc]`](data/training_data/gen_data_phi1_4a.sh) \
    $t\in[0,20]$, \
    $\Delta T=[5,10,20]$, \
    $\sigma=0.1$, \
    $N_{cells}=200$, \
    $N_{train}=[50,100,200]$, \
    $N_{valid}=N_{test}=0.2N_{train}$, \
    signal switch range: $[0.1t_{fin}, 0.15t_{fin}]$ \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$

### Binary flip landscape

- [`data_phi2_1[abc]`](data/training_data/gen_data_phi2_1a.sh) \
    $t\in[0,100]$, \
    $\Delta T=[10,50,100]$, \
    $\sigma=0.3$, \
    $N_{cells}=500$, \
    $N_{train}=[100,500,1000]$, \
    $N_{valid}=N_{test}=0.3N_{train}$, \
    signal switch range: $[0.1t_{fin}, 0.9t_{fin}]$, \
    $T_{burn}=0.05T$, \
    $s_{burn}=(-0.25, 0)$, \
    $x_0=(-1, 0)$

### Quadratic potentials

- [`data_phiq_1a`](data/training_data/gen_data_phiq_1a.sh) \
    $t\in[0,100]$, \
    $\Delta T=10$, \
    $\sigma=0.5$, \
    $N_{cells}=500$, \
    $N_{train}=100$, \
    $N_{valid}=N_{test}=0.3N_{train}$, \
    signal switch range: $[0.1t_{fin}, 0.9t_{fin}]$, \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$

- [`data_phiq_2a`](data/training_data/gen_data_phiq_2a.sh) \
    $t\in[0,100]$, \
    $\Delta T=10$, \
    $\sigma=0.1$, \
    $N_{cells}=500$, \
    $N_{train}=100$, \
    $N_{valid}=N_{test}=0.3N_{train}$, \
    signal switch range: $[0.1t_{fin}, 0.9t_{fin}]$, \
    $T_{burn}=0.1T$, \
    $s_{burn}=s|_{t=0}$, \
    $x_0=(0, -0.5)$


## Model training

We now demonstrate the model training procedure using the generated synthetic datasets.
The directories within `data/model_training_args` contain a number of tsv files, each specifying arguments to be used for an instance of model training.
We list the models trained on each of the synthetic datasets, below.
Each bullet point corresponds to a particular trained model, using the corresponding training run arguments.

### Binary choice landscape

- `data_phi1_1[abc]`

    - [run_phi1_1a_v_mmd1](data/model_training_args/synbindec/run_phi1_1a_v_mmd1.tsv)
        <!-- - [model_phi1_1a_v_mmd1_20240522_185135](data/trained_models/plnn_synbindec/model_phi1_1a_v_mmd1_20240522_185135)
        - [model_phi1_1a_v_mmd1_20240627_143649](data/trained_models/plnn_synbindec/model_phi1_1a_v_mmd1_20240627_143649)
        - [model_phi1_1a_v_mmd1_20240627_143655](data/trained_models/plnn_synbindec/model_phi1_1a_v_mmd1_20240627_143655)
        - [model_phi1_1a_v_mmd1_20240627_193208](data/trained_models/plnn_synbindec/model_phi1_1a_v_mmd1_20240627_193208)
        - [model_phi1_1a_v_mmd1_20240704_134102](data/trained_models/plnn_synbindec/model_phi1_1a_v_mmd1_20240704_134102) -->
    - [run_phi1_1b_v_mmd1](data/model_training_args/synbindec/run_phi1_1b_v_mmd1.tsv)
        <!-- - [model_phi1_1b_v_mmd1_20240802_132858](data/trained_models/plnn_synbindec/model_phi1_1b_v_mmd1_20240802_132858)  -->
    - [run_phi1_1c_v_mmd1](data/model_training_args/synbindec/run_phi1_1c_v_mmd1.tsv)
        <!-- - [model_phi1_1c_v_mmd1_20240802_132858](data/trained_models/plnn_synbindec/model_phi1_1c_v_mmd1_20240802_132858) -->
    
- `data_phi1_2[abc]`
    - [run_phi1_2a_v_mmd1](data/model_training_args/synbindec/run_phi1_2a_v_mmd1.tsv)
        <!-- - [model_phi1_2a_v_mmd1_20240807_171303](data/trained_models/plnn_synbindec/model_phi1_2a_v_mmd1_20240807_171303) 
        - [model_phi1_2a_v_mmd1_20240813_193424](data/trained_models/plnn_synbindec/model_phi1_2a_v_mmd1_20240813_193424) 
        - [model_phi1_2a_v_mmd1_20240813_194028](data/trained_models/plnn_synbindec/model_phi1_2a_v_mmd1_20240813_194028) 
        - [model_phi1_2a_v_mmd1_20240813_194433](data/trained_models/plnn_synbindec/model_phi1_2a_v_mmd1_20240813_194433) -->
    - [run_phi1_2b_v_mmd1](data/model_training_args/synbindec/run_phi1_2b_v_mmd1.tsv)
        <!-- - [model_phi1_2b_v_mmd1_20240807_171303](data/trained_models/plnn_synbindec/model_phi1_2b_v_mmd1_20240807_171303) 
        - [model_phi1_2b_v_mmd1_20240813_193441](data/trained_models/plnn_synbindec/model_phi1_2b_v_mmd1_20240813_193441) 
        - [model_phi1_2b_v_mmd1_20240813_193832](data/trained_models/plnn_synbindec/model_phi1_2b_v_mmd1_20240813_193832) 
        - [model_phi1_2b_v_mmd1_20240813_194359](data/trained_models/plnn_synbindec/model_phi1_2b_v_mmd1_20240813_194359)  -->
    - [run_phi1_2c_v_mmd1](data/model_training_args/synbindec/run_phi1_2c_v_mmd1.tsv)
        <!-- - [model_phi1_2c_v_mmd1_20240807_171303](data/trained_models/plnn_synbindec/model_phi1_2c_v_mmd1_20240807_171303) 
        - [model_phi1_2c_v_mmd1_20240813_193441](data/trained_models/plnn_synbindec/model_phi1_2c_v_mmd1_20240813_193441) 
        - [model_phi1_2c_v_mmd1_20240813_193755](data/trained_models/plnn_synbindec/model_phi1_2c_v_mmd1_20240813_193755) 
        - [model_phi1_2c_v_mmd1_20240813_194114](data/trained_models/plnn_synbindec/model_phi1_2c_v_mmd1_20240813_194114) -->

- `data_phi1_3[abc]`
    - [run_phi1_3a_v_mmd1](data/model_training_args/synbindec/run_phi1_3a_v_mmd1.tsv)
    - [run_phi1_3b_v_mmd1](data/model_training_args/synbindec/run_phi1_3b_v_mmd1.tsv)
    - [run_phi1_3c_v_mmd1](data/model_training_args/synbindec/run_phi1_3c_v_mmd1.tsv)

- `data_phi1_4[abc]`
    - [run_phi1_4a_v_mmd1](data/model_training_args/synbindec/run_phi1_4a_v_mmd1.tsv)
    - [run_phi1_4b_v_mmd1](data/model_training_args/synbindec/run_phi1_4b_v_mmd1.tsv)
    - [run_phi1_4c_v_mmd1](data/model_training_args/synbindec/run_phi1_4c_v_mmd1.tsv)


### Binary flip landscape

- `data_phi2_1[abc]`
    - [run_phi2_1a_v_mmd1](data/model_training_args/synbindec/run_phi2_1a_v_mmd1.tsv)

### Quadratic potentials

- `data_phiq_1a`
    - [run_phiq_1a_v_mmd1](data/model_training_args/synbindec/run_phiq_1a_v_mmd1.tsv)

- `data_phiq_2a`


## Model evaluation
<!-- TODO -->

# Acknowledgments
This work was inspired by the work of Sáez et al. in [Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions](https://pubmed.ncbi.nlm.nih.gov/34536382/).


# References

[1] Sáez M, Blassberg R, Camacho-Aguilar E, Siggia ED, Rand DA, Briscoe J. Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions. Cell Syst. 2022 Jan 19;13(1):12-28.e3. doi: 10.1016/j.cels.2021.08.013. Epub 2021 Sep 17. PMID: 34536382; PMCID: PMC8785827.
