from collections import defaultdict

import numpy as np

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
from torchrl.objectives import ClipPPOLoss, KLPENPPOLoss, A2CLoss, ReinforceLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
import pickle

def save(agent, test_name):
    torch.save({
            'agent' : agent          
            }, r"torch_states/%s" % test_name)
    
    file = open(r"records/%s" % test_name, 'wb')
    pickle.dump({'agent' : agent,
                 'policy' : agent.policy_module,
                 'logs' : agent.logs}
                , file)
    file.close()

class Learning_Agent(nn.Module):
    def __init__(self, seed=1, loss_mod_type='clip_ppo'):
        super().__init__()
        self.logs = defaultdict(list)        
        self.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        
        self.seed=seed
        self.loss_mod_type=loss_mod_type
        
        self.setEnv(gym_name="Spider-v0")
        
        #hyperparameters
        self.num_cells = 256  # number of cells in each layer i.e. output dim.
        self.lr = 3e-4
        self.max_grad_norm = 1.0
        
        ### PPO parameters
        self.clip_epsilon = (
            0.2  # clip value for PPO loss: see the equation in the intro for more context.
        )
        self.gamma = 0.99
        self.lmbda = 0.95
        self.entropy_eps = 1e-4
        
        
        self.initialize_learning_elements()
    
    def setEnv(self, gym_name):
        base_env = GymEnv(
            gym_name,
            device=self.device, 
            frame_skip=1
                        )
        
        self.env = TransformedEnv(
            base_env,
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(in_keys=["observation"]),
                StepCounter(),
            ),
        )
        self.env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        check_env_specs(self.env)
        
    def select_loss(self, type):        
        
        if type == 'clip_ppo':
            loss_module = ClipPPOLoss(
                actor=self.policy_module,
                critic=self.value_module,
                advantage_key="advantage",
                clip_epsilon=self.clip_epsilon,
                entropy_bonus=bool(self.entropy_eps),
                entropy_coef=self.entropy_eps,
                value_target_key=self.advantage_module.value_target_key,
                critic_coef=1.0,
                gamma=self.gamma,
                loss_critic_type="smooth_l1",
            )
        elif type == 'KL':
            loss_module = KLPENPPOLoss(
                actor=self.policy_module,
                critic=self.value_module,
                beta=.8,
                advantage_key="advantage",
                entropy_bonus=bool(self.entropy_eps),
                entropy_coef=self.entropy_eps,                
                value_target_key=self.advantage_module.value_target_key,
                critic_coef=1.0,
                loss_critic_type="smooth_l1",
            )
        elif type == 'A2C':
            loss_module = A2CLoss(
                actor=self.policy_module,
                critic=self.value_module,                
                # advantage_key="advantage",
                # entropy_bonus=bool(self.entropy_eps),
                # entropy_coef=self.entropy_eps,                
                # value_target_key=self.advantage_module.value_target_key,
                # critic_coef=0.2,
                # loss_critic_type="l2",
            )
        elif type == 'REINFORCE':
            loss_module = ReinforceLoss(
                actor=self.policy_module,
                critic=self.value_module, 
                # loss_critic_type="smooth_l1",                               
                # advantage_key="advantage",
                # value_target_key=self.advantage_module.value_target_key,
            )
            
        return loss_module
            
        
    def initialize_learning_elements(self):
        actor_net = nn.Sequential(
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(2 * self.env.action_spec.shape[-1], device=self.device),
            NormalParamExtractor(),
        )        
        
        value_net = nn.Sequential(
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(1, device=self.device),
        )

        self.value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )
        
        ### POLICY
        self.policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        self.policy_module = ProbabilisticActor(
            module=self.policy_module,
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": self.env.action_spec.space.minimum,
                "max": self.env.action_spec.space.maximum,
            },
            return_log_prob=True,
        )
        
        self.policy_module(self.env.reset(seed=self.seed))
        self.value_module(self.env.reset(seed=self.seed))
                       
        ### LOSS
        self.advantage_module = GAE(
            gamma=self.gamma, lmbda=self.lmbda, value_network=self.value_module, average_gae=True
        )
        
        self.loss_module = self.select_loss(self.loss_mod_type)
        
        self.optim = torch.optim.SGD(self.loss_module.parameters(), lr=self.lr, momentum=0.9)
           
    
    def train(self, num_epochs, total_frames, frame_skip, frames_per_batch, sub_batch_size):
        
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )
        
        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=self.device,
        )
        
        self.logs = defaultdict(list)
        pbar = tqdm(total=total_frames * frame_skip)
        eval_str = ""
        
        for i, tensordict_data in enumerate(self.collector):
            for _ in range(num_epochs):
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = self.replay_buffer.sample(sub_batch_size)
                    if self.loss_mod_type=='KL':
                        pass
                    else:
                        loss_vals = self.loss_module(subdata.to(self.device))
                    
                    if self.loss_mod_type=='REINFORCE':
                        loss_value = (
                        loss_vals["loss_actor"]
                        + loss_vals["loss_value"]
                    )
                    else:
                        loss_value = (
                            loss_vals["loss_objective"]
                            + loss_vals["loss_critic"]
                            + loss_vals["loss_entropy"]
                        )
                    loss_value.backward()
                    
                    self.optim.step()
                    self.optim.zero_grad()

            self.logs["reward"].append(tensordict_data["next", "reward"].mean().item())    
            self.logs["min_reward"].append(tensordict_data["next", "reward"].min().item())
            self.logs["max_reward"].append(tensordict_data["next", "reward"].max().item())
    
            pbar.update(tensordict_data.numel() * frame_skip)
            cum_reward_str = (
                f"average reward={self.logs['reward'][-1]: 4.4f} (init={self.logs['reward'][0]: 4.4f})"
            )
            self.logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {self.logs['step_count'][-1]}"
            self.logs["lr"].append(self.optim.param_groups[0]["lr"])
            lr_str = f"lr policy: {self.logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                eval_str = self.test()
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    
    def test(self):
        with set_exploration_mode("mean"), torch.no_grad():
            
            eval_rollout = self.env.rollout(1000, self.policy_module)
            
            self.logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            self.logs["min eval reward"].append(eval_rollout["next", "reward"].min().item())
            self.logs["max eval reward"].append(eval_rollout["next", "reward"].max().item())
            
            self.logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            
            self.logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {self.logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {self.logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {self.logs['eval step_count'][-1]}"
            )
            del eval_rollout
            return eval_str
        
    def plot_logs(self):
        plt.figure(figsize=(12, 5.5))
        plt.subplot(1, 3, 1)
        plt.fill_between(
            x = range(len(self.logs['reward'])),
            y1=self.logs["min_reward"], 
            y2=self.logs["max_reward"], 
            alpha=.5
            )
        plt.plot(self.logs["reward"])
        plt.title("training rewards (average)", weight='semibold')

        plt.subplot(1, 3, 2)
        plt.fill_between(
            x = range(len(self.logs['eval reward'])),
            y1=self.logs["min eval reward"], 
            y2=self.logs["max eval reward"], 
            alpha=.5
            )
        plt.plot(self.logs["eval reward"])
        plt.title("test rewards (average)", weight='semibold')


        plt.subplot(1, 3, 3)
        plt.plot(self.logs["eval reward (sum)"])
        plt.title("Return (test)", weight='semibold')

        plt.show()
        
        
        
if __name__ == "__main__":
    
    #Load Previous Agent
    # load_contents = torch.load(r"torch_states/%s" % 'test_121923')
    # agent = load_contents['agent']
    
    # out = pickle.load(r"records/%s" % 'test_121923_2')
    # agent = out['agent']
        
    seeds = [10,20,30,40,50]
    # loss_types = ['KL']
    loss_types = ['REINFORCE']
    agent_logs = []
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axar = plt.subplots(1,3, figsize=(12,5.5))
    axar[0].set_title("Training Rewards (average)", weight='semibold')
    axar[1].set_title("Test Rewards (average)", weight='semibold')
    axar[2].set_title("Return (test)", weight='semibold')
    
    for loss_type in loss_types:
        plot_logs = []
        for seed in seeds:    
            agent = Learning_Agent(seed=seed,
                                   loss_mod_type=loss_type)
            agent.train(
                        # num_epochs = 10, 
                        # total_frames = 1_000, 
                        # frame_skip=1, 
                        # frames_per_batch= 10, 
                        # sub_batch_size = 64,
                        
                        num_epochs = 20, 
                        total_frames = 200_000, 
                        frame_skip=1, 
                        frames_per_batch= 5_000, 
                        sub_batch_size = 64,                        
                        )
            save(agent, 'test_121923_%s_%i' % (loss_type, seed))
            agent_logs.append(agent.logs)
            plot_logs.append(agent.logs)
        
        for i, ax_param in enumerate(['reward','eval reward','eval reward (sum)']):
            x_low = np.minimum.reduce([log[ax_param] for log in plot_logs])
            x_med = np.median([log[ax_param] for log in plot_logs], axis=0)
            x_high = np.maximum.reduce([log[ax_param] for log in plot_logs])    
            axar[i].fill_between(list(range(len(x_low))), 
                                 y1=x_low, 
                                 y2=x_high, 
                                 alpha=.5,
                                 label='%s (seed min, max)' % loss_type)
            axar[i].plot(x_med, label=loss_type)
        
        
        # for log in agent_logs:
        #     axar[0].plot(log['reward'])        
        #     axar[1].plot(log['eval reward'])
        #     axar[2].plot(log['eval reward (sum)'])
    
    axar[2].legend(bbox_to_anchor=(1,1))
    fig.tight_layout()
    fig.savefig(r"output.png")
    
    file = open(r"plot_relevant%s" % 'final', 'wb')
    pickle.dump({
        'seeds' : seeds,
        'loss_types' : loss_types,
        'logs' : agent_logs,
    }
                , file)
    file.close()

    
    plt.show()
    print('DONE')
    
    
    