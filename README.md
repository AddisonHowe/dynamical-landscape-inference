# dynamical-landscape-inference

This repository demonstrates the application of the PLNN architecture described in "[Dynamical systems theory informed learning of cellular differentiation landscapes](https://www.biorxiv.org/content/10.1101/2024.09.21.614191v1)" and implemented in a separate Github repository: [https://github.com/AddisonHowe/plnn](https://github.com/AddisonHowe/plnn).


## Setup
Basic setup, without GPU acceleration:
```bash
conda create -p ./env python=3.9 jax=0.4 numpy=1.26 matplotlib=3.8 scikit-learn=1.5 pytorch=2.0 torchvision equinox=0.11 optax=0.1 pyyaml=6.0 tqdm ipykernel pytest
conda activate env
pip install diffrax==0.6.0
```

For GPU support:
```bash
conda create -p ./env python=3.9 numpy=1.25 matplotlib=3.8 scikit-learn=1.5 pytest=7.4 cuda-compat=12.4 pyyaml=6.0 tqdm ipykernel ipywidgets --yes
conda activate env
pip install --upgrade pip
pip install jax[cuda12] optax==0.1.7 diffrax==0.6.0 equinox==0.11.5 torch==2.0.1 torchvision torchaudio
```

Then, install the PLNN project at [https://github.com/AddisonHowe/plnn](https://github.com/AddisonHowe/plnn):
```bash
pip install git+https://github.com/AddisonHowe/plnn.git@v0.1.0-alpha
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
After the training process, we evaluate the resulting model using the notebooks contained in `notebooks/model_evaluation/`.
The bash scripts located in `scripting/model_evaluation/` automate this process, and can be run as follows:

```bash
# Evaluate models listed in...

# scripting/model_evaluation/arglist_nb_eval_model_plnn_synbindec.tsv
sh scripting/model_evaluation/run_all_nb_eval_model_plnn_synbindec.sh

# scripting/model_evaluation/arglist_nb_eval_model_plnn_quadratic.tsv
sh scripting/model_evaluation/run_all_nb_eval_model_plnn_quadratic.sh

# scripting/model_evaluation/arglist_nb_eval_model_alg_synbindec.tsv
sh scripting/model_evaluation/run_all_nb_eval_model_alg_synbindec.sh

# scripting/model_evaluation/arglist_nb_eval_model_alg_quadratic.tsv
sh scripting/model_evaluation/run_all_nb_eval_model_alg_quadratic.sh
```

These commands will run the model evaluation notebook on all models listed in the corresponding arglist file, and generate output in the directory `data/model_evaluation/`.

### Automated figure generation
The images generated in the evaluation process can be arranged into a set of pdfs in an automated procedure, using the scripts available in the `scripting/autofig/` directory.
First, create or modify an environment file `.env`, and set the following environment variables:
```bash
# Filename: dynamical-landscape-inference/.env
ILLUSTRATOR_PATH=<path/to/adobe/illustrator.app>
PROJ_DIR_TILDE=<~/path/to/dynamical-landscape-inference/>  # using tilde for home directory
```
Then, run the figure generation scripts:
```bash
sh scripting/autofig/alg_quadratic/generate_ai_files_from_template.sh
sh scripting/autofig/alg_synbindec/generate_ai_files_from_template.sh
sh scripting/autofig/facs/generate_ai_files_from_template.sh
sh scripting/autofig/plnn_quadratic/generate_ai_files_from_template.sh
sh scripting/autofig/plnn_synbindec/generate_ai_files_from_template.sh
```

# Figure generation
The figures in the manuscript and supplement can be reproduced using the scripts available in the `figures/` directory.
In `figures/manuscript/` there are a number of python scripts that generate the individual plots appearing in the main figures of the manuscript.
The corresponding shell scripts run these python files with any hyperparameters specified.
Note that in order to generate the plots for Figures 6 and 7, one must set the environment variable `MESC_PROJ_PATH` in the file `.env` to point to the `mescs-invitro-facs` project directory (available at [https://github.com/AddisonHowe/mescs-invitro-facs](https://github.com/AddisonHowe/mescs-invitro-facs)) containing the FACS data used in those plots.

```bash
# Generate all plots appearing in primary figures
sh figures/make_all_manuscript_plots.sh

# The command above runs the following scripts:
# sh figures/manuscript/run_make_fig1_landscape_models.sh
# sh figures/manuscript/run_make_fig3_synthetic_training.sh
# sh figures/manuscript/run_make_fig4_sampling_rate_sensitivity.sh
# sh figures/manuscript/run_make_fig5_dimred_schematic.sh
# export MESC_PROJ_PATH=<path/to/mesc-invitro-facs>
# sh figures/manuscript/run_make_fig6_facs_training.sh
# sh figures/manuscript/run_make_fig7_facs_evaluation.sh
```
A similar set of commands are available to generate plots appearing in the supplementary information.
```bash
# Generate all plots appearing in supplemental figures
sh figures/make_all_supplement_plots.sh

# The command above runs the following scripts:
# sh run_make_figure_s1
```

# Acknowledgments
This work was inspired by the work of Sáez et al. in [Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions](https://pubmed.ncbi.nlm.nih.gov/34536382/).


# References
[1] Sáez M, Blassberg R, Camacho-Aguilar E, Siggia ED, Rand DA, Briscoe J. Statistically derived geometrical landscapes capture principles of decision-making dynamics during cell fate transitions. Cell Syst. 2022 Jan 19;13(1):12-28.e3. doi: 10.1016/j.cels.2021.08.013. Epub 2021 Sep 17. PMID: 34536382; PMCID: PMC8785827.
