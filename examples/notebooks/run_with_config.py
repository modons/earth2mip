import argparse
import logging
import os
import sys
import xarray
import cftime
import json
#
import earth2mip.networks.dlwp as dlwp
import earth2mip.networks.pangu as pangu
import os
from earth2mip import registry, inference_ensemble
from earth2mip.initial_conditions import cds,rda
from earth2mip.schema import EnsembleRun, Grid, PerturbationStrategy
from earth2mip.networks import get_model, Inference
import os, json, logging, datetime
import numpy as np
from earth2mip.inference_ensemble import get_initializer,run_inference
from modulus.distributed.manager import DistributedManager
import torch

logging.basicConfig(level=logging.INFO)

config = './test_config.json'

if config is None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--weather_model", default=None)
    args = parser.parse_args()
    config = args.config

# If config is a file
if os.path.exists(config):
    config: EnsembleRun = EnsembleRun.parse_file(config)
# If string, assume JSON string
elif isinstance(config, str):
    config: EnsembleRun = EnsembleRun.parse_obj(json.loads(config))
# Otherwise assume parsable obj
else:
    raise ValueError(
        f"Passed config parameter {config} should be valid file or JSON string"
    )

print(config)

# Set up parallel
DistributedManager.initialize()
device = DistributedManager().device
group = torch.distributed.group.WORLD

logging.info(f"Earth-2 MIP config loaded {config}")
logging.info(f"Loading model onto device {device}")
model = get_model(config.weather_model, device=device)
logging.info(f"Constructing initializer data source")
perturb = get_initializer(
    model,
    config,
)
logging.info(f"Running inference")
run_inference(model, config, perturb, group)

