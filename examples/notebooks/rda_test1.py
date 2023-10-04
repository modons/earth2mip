import earth2mip.networks.dlwp as dlwp
import earth2mip.networks.pangu as pangu
import os
from earth2mip import registry, inference_ensemble
from earth2mip.initial_conditions import cds,rda
import os, json, logging, datetime


model_registry = os.path.join(os.path.dirname(os.path.realpath(os. getcwd())), "/glade/u/home/zilumeng/earth2mip/examples/models")
os.makedirs(model_registry, exist_ok=True)
pangu_registry = os.path.join(model_registry, "/glade/u/home/zilumeng/earth2mip/examples/models/pangu/")

# with open(os.path.join(pangu_registry, 'metadata.json'), 'w') as outfile:
#     json.dump({"entrypoint": {"name": "earth2mip.networks.pangu:load"}}, outfile, indent=2)
# pangu_registry = os.path.join(model_registry, "pangu")

# Load DLWP model from registry
# package = registry.get_model("dlwp")
# dlwp_inference_model = dlwp.load(package)

# Load Pangu model(s) from registry
package = registry.get_model("/glade/u/home/zilumeng/earth2mip/examples/models/pangu/")
pangu_inference_model = pangu.load(package)
pangu_data_source = rda.DataSource(pangu_inference_model.in_channel_names)
time = datetime.datetime(2018, 1, 1)
pangu_ds = inference_ensemble.run_basic_inference(
    pangu_inference_model,
    n=24, # Note we run 24 steps here because Pangu is at 6 hour dt
    data_source=pangu_data_source,
    time=time,
)
print(pangu_ds)
