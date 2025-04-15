#%%
import os,sys
import numpy as np
from sklearn.model_selection import train_test_split
from nequip.data import ASEDataset
import random
from ase.io import read,write

from nequip.utils import Config
from nequip.model import model_from_config
from nequip.data import dataset_from_config

import logging

mode_train = "charge"
# mode_train = "raw"

if mode_train == "raw":
    root = '../train_raw'
    run_name = 'raw_train'
elif mode_train == "charge":
    root = '../train_charge'
    run_name = 'charge_train'
    
    
default_config = dict(
    root=root,
    run_name=run_name,
    wandb=True,
    wandb_project="bader charge",
    model_builders=[
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "ForceOutput",
        "RescaleEnergyEtc",
    ],
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
    _jit_fusion_strategy=[("DYNAMIC", 3)],
)



def main():
    #%%
    random.seed(0)

    # num_training=int(sys.argv[1])
    ext_xyz = read('/home/users/loewanc6/storage/script_Phd/bader_charge/train_stuff/data/merge_energy.xyz',':')
    config=Config.from_file("config_base.yaml", defaults=default_config)
    if mode_train == "raw":
        config["root"]=root
        config["run_name"]=run_name
    
    if mode_train == "charge":
        config["root"]=root
        config["run_name"]=run_name
        config["model_builders"][1]="chargeModel" # replace energymodel
        config["train_on_keys"].append("bader") # add bader charge to the training keys
    # Create a list of indices
    indices = np.arange(len(ext_xyz))

    # Split indices into training and validation sets

    validation_indices=random.sample(list(indices), int(len(ext_xyz) * 0.1))
    test_indices=random.sample(list(indices), int(len(ext_xyz) * 0.1))
    train_indices=[i for i in indices if i not in validation_indices and i not in test_indices]

    # Use indices to get training and validation data
    validation_data = [ext_xyz[i] for i in validation_indices]
    test_data = [ext_xyz[i] for i in test_indices]

    train_data = [ext_xyz[i] for i in train_indices]

    # Update the trainer with the new datasets
    config['n_train']=len(train_data)
    config['n_val']=len(validation_data)

    # make a list of chemicals symbols unique
    symbols = set()
    for atoms in train_data:
        symbols.update(atoms.get_chemical_symbols())
    for atoms in validation_data:
        symbols.update(atoms.get_chemical_symbols())
    for atoms in test_data:
        symbols.update(atoms.get_chemical_symbols())
    symbols = list(symbols)
    #%%

    import wandb  # noqa: F401
    from nequip.train.trainer_wandb import TrainerWandB
    from nequip.utils.wandb import init_n_update

    config = init_n_update(config)

    trainer = TrainerWandB(model=None, **dict(config))

    config.update(trainer.params)
    path_tmp_train='../data/data_train.extxyz'
    path_tmp_eval='../data/data_val.extxyz'
    write(path_tmp_train,train_data)
    write(path_tmp_eval,validation_data)
    config['dataset_file_name']=path_tmp_train
    config['validation_dataset']='ase'
    config['validation_dataset_file_name']=path_tmp_eval
    config["dataset"] = "ASEDataset"
    config["chemical_symbols"] = symbols
    config.save("config_script.yaml")
    #%%
    dataset = dataset_from_config(config,prefix="dataset")
    validation_dataset = dataset_from_config(config, prefix='validation_dataset')
    # os.remove(path_tmp_eval)
    # os.remove(path_tmp_train)

    trainer.set_dataset(dataset, validation_dataset)

    # = Build model =
    final_model = model_from_config(
        config=config, initialize=True, dataset=trainer.dataset_train
    )
    trainer.model = final_model
    trainer.update_kwargs(config)
    logging.info("Successfully built the network...")
    trainer.save()
    trainer.train()

#%%
if __name__ == '__main__':
    main()
