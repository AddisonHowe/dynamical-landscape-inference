# Model training argument files

### Synthetic binary decision landscapes

<!-- REPLACEMENT START KEY [SYNBINDEC] -->
| argfile | training data | nepochs | patience | batch size | phi layers | ncells | sigma | loss | solver | dt0 | dt scheduling | learning rate | optimizer |
 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
 | [model_phi1_1a_v_mmd1](../model_training_args/synbindec/run_phi1_1a_v_mmd1.tsv) | [training data](../training_data/data_phi1_1a/training)<br>[validation data](../training_data/data_phi1_1a/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_1b_v_mmd1](../model_training_args/synbindec/run_phi1_1b_v_mmd1.tsv) | [training data](../training_data/data_phi1_1b/training)<br>[validation data](../training_data/data_phi1_1b/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_1c_v_mmd1](../model_training_args/synbindec/run_phi1_1c_v_mmd1.tsv) | [training data](../training_data/data_phi1_1c/training)<br>[validation data](../training_data/data_phi1_1c/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_2a_v_mmd1](../model_training_args/synbindec/run_phi1_2a_v_mmd1.tsv) | [training data](../training_data/data_phi1_2a/training)<br>[validation data](../training_data/data_phi1_2a/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_2b_v_mmd1](../model_training_args/synbindec/run_phi1_2b_v_mmd1.tsv) | [training data](../training_data/data_phi1_2b/training)<br>[validation data](../training_data/data_phi1_2b/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_2c_v_mmd1](../model_training_args/synbindec/run_phi1_2c_v_mmd1.tsv) | [training data](../training_data/data_phi1_2c/training)<br>[validation data](../training_data/data_phi1_2c/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_3a_v_mmd1](../model_training_args/synbindec/run_phi1_3a_v_mmd1.tsv) | [training data](../training_data/data_phi1_3a/training)<br>[validation data](../training_data/data_phi1_3a/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_3b_v_mmd1](../model_training_args/synbindec/run_phi1_3b_v_mmd1.tsv) | [training data](../training_data/data_phi1_3b/training)<br>[validation data](../training_data/data_phi1_3b/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_3c_v_mmd1](../model_training_args/synbindec/run_phi1_3c_v_mmd1.tsv) | [training data](../training_data/data_phi1_3c/training)<br>[validation data](../training_data/data_phi1_3c/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_4a_v_mmd1](../model_training_args/synbindec/run_phi1_4a_v_mmd1.tsv) | [training data](../training_data/data_phi1_4a/training)<br>[validation data](../training_data/data_phi1_4a/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_4b_v_mmd1](../model_training_args/synbindec/run_phi1_4b_v_mmd1.tsv) | [training data](../training_data/data_phi1_4b/training)<br>[validation data](../training_data/data_phi1_4b/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi1_4c_v_mmd1](../model_training_args/synbindec/run_phi1_4c_v_mmd1.tsv) | [training data](../training_data/data_phi1_4c/training)<br>[validation data](../training_data/data_phi1_4c/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi2_1a_v_mmd1](../model_training_args/synbindec/run_phi2_1a_v_mmd1.tsv) | [training data](../training_data/data_phi2_1a/training)<br>[validation data](../training_data/data_phi2_1a/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi2_1b_v_mmd1](../model_training_args/synbindec/run_phi2_1b_v_mmd1.tsv) | [training data](../training_data/data_phi2_1b/training)<br>[validation data](../training_data/data_phi2_1b/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phi2_1c_v_mmd1](../model_training_args/synbindec/run_phi2_1c_v_mmd1.tsv) | [training data](../training_data/data_phi2_1c/training)<br>[validation data](../training_data/data_phi2_1c/validation) | 2000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [200 500 1000]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |

<!-- REPLACEMENT END KEY [SYNBINDEC] -->


### Quadratic Landscapes

<!-- REPLACEMENT START KEY [QUADRATIC] -->
| argfile | training data | nepochs | patience | batch size | phi layers | ncells | sigma | loss | solver | dt0 | dt scheduling | learning rate | optimizer |
 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
 | [model_phiq_1a_v_mmd1](../model_training_args/quadratic/run_phiq_1a_v_mmd1.tsv) | [training data](../training_data/data_phiq_1a/training)<br>[validation data](../training_data/data_phiq_1a/validation) | 1000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [100 250 500]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |
 | [model_phiq_2a_v_mmd1](../model_training_args/quadratic/run_phiq_2a_v_mmd1.tsv) | [training data](../training_data/data_phiq_2a/training)<br>[validation data](../training_data/data_phiq_2a/validation) | 1000 | 100 | 250 | 16 32 32 16 | 200 | 0.05 | mmd | heun | 1e-1 | stepped <br>bounds: [100 250 500]<br>scales: [0.5 0.5 0.5] | exponential_decay <br>(1e-2, 1e-5, 50) | rms <br>m=0.5<br>decay=0.9<br>clip=1.0 |

<!-- REPLACEMENT END KEY [QUADRATIC] -->


### Algebraic Binary Decision Landscapes

<!-- REPLACEMENT START KEY [ALGBINDEC] -->
| argfile | training data | nepochs | patience | batch size | phi layers | ncells | sigma | loss | solver | dt0 | dt scheduling | learning rate | optimizer |
 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

<!-- REPLACEMENT END KEY [MISC] -->

### FACS Landscapes

<!-- REPLACEMENT START KEY [FACS] -->
| argfile | training data | nepochs | patience | batch size | phi layers | ncells | sigma | loss | solver | dt0 | dt scheduling | learning rate | optimizer |
 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

<!-- REPLACEMENT END KEY [FACS] -->


### Misc. Landscapes

<!-- REPLACEMENT START KEY [MISC] -->
| argfile | training data | nepochs | patience | batch size | phi layers | ncells | sigma | loss | solver | dt0 | dt scheduling | learning rate | optimizer |
 | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

<!-- REPLACEMENT END KEY [MISC] -->
