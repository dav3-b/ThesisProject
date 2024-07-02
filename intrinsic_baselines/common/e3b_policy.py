import copy
import random

import numpy as np
import torch
from gym.spaces.box import Box
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
#from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder, LSTMStateEncoder, GRUStateEncoder
from habitat_extensions.rnn_state_encoder_custom import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Policy, Net 
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
#from habitat_baselines.rl.ddppo.policy.resnet import resnet18
from intrinsic_baselines.common.models import FwdStateEncoder, resnet18
from torch import nn
from torchvision.models import resnet50

from intrinsic_baselines.common.models import SimpleCNN, Flatten
#from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder

import clip

class E3BBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self,
                 observation_space,
                 obs_resized_dim,
                 num_concat_obs,
                 action_space,
                 hidden_size,
                 #pretrained_visual_encoder,
                 visual_encoder_type="simplecnn",
                 state_encoder_type="rnn",
                 use_rnn=True
                ):
        super().__init__()
        self._hidden_size = hidden_size
        self._state_encoder_type = state_encoder_type.lower()
        self._visual_encoder_type = visual_encoder_type.lower()
        self._use_rnn = use_rnn
        #self._pret_vis_enc = pretrained_visual_encoder

        # Changed initial size of input to cnn
        cnn_space = copy.deepcopy(observation_space)
        if 'rgb' in observation_space.spaces.keys():
            cnn_space.spaces['rgb'] = Box(low=0, high=255,
                                          shape=(
                                                obs_resized_dim,
                                                obs_resized_dim,
                                                num_concat_obs
                                                ),
                                          dtype='uint8')
        if 'depth' in observation_space.spaces.keys():
            cnn_space.spaces['depth'] = Box(low=0, high=255,
                                            shape=(
                                                obs_resized_dim,
                                                obs_resized_dim,
                                                num_concat_obs
                                                ),
                                            dtype='float32')
        #if 'rgb' in observation_space.spaces.keys():
        #    cnn_space.spaces['rgb'] = Box(low=0, high=255, shape=(42, 42, 4),
        #                                  dtype='uint8')
        #if 'depth' in observation_space.spaces.keys():
        #    cnn_space.spaces['depth'] = Box(low=0, high=255, shape=(42, 42, 4),
        #                                    dtype='float32')
        if self._visual_encoder_type == "simplecnn":
            self.visual_encoder = SimpleCNN(cnn_space, hidden_size)
        elif self._visual_encoder_type == "resnet":
            self.visual_encoder = resnet18(in_channels=num_concat_obs*2)
            #self.visual_encoder = resnet18(in_channels=2, base_planes=16, ngroups=16//2)
            #self.visual_encoder = ResNetEncoder(
            #    cnn_space,
            #	baseplanes=32,
            #	ngroups=32//2,
            #	make_backbone=getattr(resnet, "resnet18"),
            #	normalize_visual_inputs="rgb" in cnn_space.spaces,
            #    #output_dim=512
			#)
            #self.join = nn.Sequential(
            #    nn.Flatten(),
            #    nn.Linear(
            #        np.prod(self.visual_encoder.output_shape), self._hidden_size
            #    ),
            #    nn.ReLU(True),
            #)
        #elif self._visual_encoder_type == "clip":
        #    self.visual_encoder, preprocess = clip.load("RN50")
        #    #for param in self.visual_encoder.parameters():
        #    #    param.requires_grad = False
        else:
            raise RuntimeError(f"Did not recognize visual encoder type '{self._visual_encoder_type}'")

        if self._state_encoder_type == "rnn":
            if use_rnn:
                self.state_encoder = RNNStateEncoder(
                    #(0 if self.is_blind else self._hidden_size),
                    self._hidden_size,
                    self._hidden_size,
                    rnn_type="LSTM"
                )
                #self.state_encoder = build_rnn_state_encoder(
                #    (0 if self.is_blind else self._hidden_size),
                #    self._hidden_size,
                #    rnn_type='lstm',
                #)
            else:
                raise RuntimeError(f"The flag use_rnn is '{use_rnn}'")
        elif self._state_encoder_type == "custom":
            self.state_encoder = nn.Sequential(
                nn.Flatten(),
                #nn.Linear(
                #    np.prod(self.visual_encoder.output_shape), self._hidden_size
                #),
                #nn.ReLU(True),
                nn.Linear(self._hidden_size, self._hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self._hidden_size, self._hidden_size), 
                nn.ReLU(inplace=True),
                nn.Linear(self._hidden_size, self._hidden_size)
            )
        elif self._state_encoder_type == "fwd":
            self.state_encoder = FwdStateEncoder(self._hidden_size, self._hidden_size // 2)
        else:
            raise RuntimeError(f"Did not recognize state encoder type '{self._state_encoder_type}'")
        
        #if self._state_encoder_type == "lstm":
            #self.state_encoder = nn.LSTM(
            #    (0 if self.is_blind else self._hidden_size),
            #    self._hidden_size
            #    #num_layers=num_layers,
            #)
        #elif self._state_encoder_type == "gru":
            #self.state_encoder = nn.GRU(
            #    (0 if self.is_blind else self._hidden_size),
            #    self._hidden_size
            #    #num_layers=num_layers,
            #)
        
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
    #    return self.visual_encoder.is_blind
        pass

    @property
    def num_recurrent_layers(self):
        #if self._state_encoder_type == "custom":
        #    return None
        #else:
        return self.state_encoder.num_recurrent_layers

    def _preprocessing(self, observations):
        obs_input = []

        if "rgb" in observations:
            n_input_rgb = observations["rgb"].shape[2]
        else:
            n_input_rgb = 0

        if "depth" in observations:
            n_input_depth = observations["depth"].shape[2]
        else:
            n_input_depth = 0
        
        if n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            obs_input.append(rgb_observations)

        if n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            obs_input.append(depth_observations)

        #if self.obs_transform:
        #    cnn_input = [self.obs_transform(inp) for inp in cnn_input]

        obs_input = torch.cat(obs_input, dim=1)

        return obs_input

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, flag=False):
        features = {}
        
        #if not self.is_blind:
            #if self._pret_vis_enc and self._visual_encoder_type == "clip":
            #    with torch.no_grad():
            #        obs = self._preprocessing(observations)
            #        perception_embed = self.visual_encoder.encode_image(obs)
            #        x = [perception_embed]
            #else:
        if self._visual_encoder_type == "resnet":
            observations = self._preprocessing(observations)
        perception_embed = self.visual_encoder(observations)
        #if self._visual_encoder_type == "resnet":
        #    perception_embed = self.join(perception_embed)
        #    if flag:
        #        features['join'] = perception_embed.detach()
        x = [perception_embed]
        #else:
        #    blind_observation = torch.zeros_like(observations)
        #    perception_embed = self.visual_encoder(blind_observation)
        #    x = [perception_embed]

        x = torch.cat(x, dim=1)
        if flag:
            features['visual_encoder'] = x.detach()

        if self._use_rnn:
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks.to(torch.bool))
            if flag:
                features['state_encoder'] = x.detach()
        #elif self._state_encoder_type == "fwd":
        #    x = self.state_encoder()
        else:
            rnn_hidden_states = None
            x = self.state_encoder(x)
            if flag:
                features['state_encoder'] = x.detach()

        if flag:
            return x, rnn_hidden_states, features
        else:
            return x, rnn_hidden_states

    def to_ddp(self, device_ids, output_device):
        self.pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        #self.state_encoder.rnn = nn.parallel.DistributedDataParallel(self.state_encoder.rnn, device_ids=device_ids,
        #                                                             output_device=output_device,
        #                                                             process_group=self.pg1)
        if self._state_encoder_type != "rnn":
            self.state_encoder = nn.parallel.DistributedDataParallel(self.state_encoder, device_ids=device_ids,
                                                                     output_device=output_device,
                                                                     process_group=self.pg1)