import pickle  

from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, set_exploration_mode
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

# open a file, where you stored the pickled data
record = r"records/20epcs_1mil_256cells_1kframs_1fskip_motor_contractForce_minuszang_more_fric"
record = r"records/20epcs_1mil_256cells_1kframs_1fskip_motormix_contractForce_minuszang_more_fric"
record = r"records/20epcs_5mil_256cells_1kframs_1fskip_motor_contractForce_minuszang_more_fric"
record = r"records/20epcs_20mil_256cells_1kframs_1fskip_motor_contractForce_minuszang_more_fric"
record = r"records/40mil"
file = open(record, 'rb')

# dump information to that file
out = pickle.load(file)
if type(out) == type({}):
    policy_module = out['policy']
    logs = out['logs']
else:
    policy_module = out

# close the file
file.close()

device = "cpu" if not torch.cuda.is_available() else "cuda:0"
frame_skip = 1

r_env = GymEnv(
    "Spider-v0",
    device=device, 
    frame_skip=frame_skip,
    render_mode="human"
                  )

rr_env = TransformedEnv(
    r_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(in_keys=["observation"]),
        StepCounter(),
    ),
)

rr_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
rollout = rr_env.rollout(1000, policy_module)

print(rollout)

data = rollout["observation"].cpu().detach()
x,y,z = data[:][-13:-9]
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x.tolist(),y.tolist(),z.tolist())
plt.show()