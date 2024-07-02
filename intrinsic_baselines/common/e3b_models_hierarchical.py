import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from gym.spaces.box import Box
#from intrinsic_baselines.common.models import GatedPixelCNN
from intrinsic_baselines.common.e3b_policy import E3BBaselineNet
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy, PointNavResNetNet

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

EPS_PPO = 1e-5

class E3BModels(nn.Module):
    def __init__(
            self,
            clip_param,
            update_epochs,
            obs_resized_dim,
            num_concat_obs,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr=None,
            eps=None,
            max_grad_norm=None,
            use_clipped_value_loss=True,
            use_normalized_advantage=True,
            observation_spaces=None,
            fwd_model=None,
            inv_model=None,
            impact_hidden_size=512,
            paper_encoder=False,
            pretrained_paper_encoder=False,
            pretrained_paper_encoder_weights=None,
            action_space=None,
            config=None,
            #pretrained_visual_encoder=False,
            use_rnn=True,
            device='cpu',
            use_ddp=False,
    ):

        super().__init__()

        self.clip_param = clip_param
        self.update_epochs = update_epochs
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.curiosity_beta = 0.2
        
        self.paper_encoder = paper_encoder  

        self.config = config

        self._counter = 0

        if paper_encoder:
            obs_space = copy.deepcopy(observation_spaces)
            if 'rgb' in observation_spaces.spaces.keys():
                obs_space.spaces['rgb'] = Box(low=0, high=255,
                                            shape=(
                                                    obs_resized_dim,
                                                    obs_resized_dim,
                                                    num_concat_obs
                                                    ),
                                            dtype='uint8')
            if 'depth' in observation_spaces.spaces.keys():
                obs_space.spaces['depth'] = Box(low=0, high=255,
                                                shape=(
                                                    obs_resized_dim,
                                                    obs_resized_dim,
                                                    num_concat_obs
                                                    ),
                                                dtype='float32')

            #obs_space2 = copy.deepcopy(observation_spaces)
            #obs_transforms = get_active_obs_transforms(self.config)
            #obs_space2 = apply_obs_transforms_obs_space(obs_space2, obs_transforms)
            

            if pretrained_paper_encoder:
                actions = action_space
                num_recurrent_layers = 2
                self.rnn_type = "LSTM"
                backbone = "resnet50"
            else:
                actions = None
                num_recurrent_layers = 0
                self.rnn_type = "none"
                backbone = "resnet18"
                #actions = action_space
                #num_recurrent_layers = 2
                #self.rnn_type = "LSTM"
                #backbone = "resnet18"

            self.obs_encoder = PointNavResNetNet(observation_space=obs_space,
                                                action_space=actions,
                                                hidden_size=impact_hidden_size,
                                                num_recurrent_layers=num_recurrent_layers,
                                                rnn_type=self.rnn_type,
                                                backbone=backbone, 
                                                normalize_visual_inputs="rgb" in obs_space.spaces,
                                                resnet_baseplanes=32,
                                                force_blind_policy=False)
            
            if pretrained_paper_encoder and pretrained_paper_encoder_weights is not None:
                from habitat_baselines.common.baseline_registry import baseline_registry
                policy = baseline_registry.get_policy("PointNavResNetPolicy")
                self.actor_critic = policy.from_config(None, obs_space, action_space)

                #self.obs_encoder = PointNavResNetPolicy(
                #    observation_space=obs_space,
                #    action_space=actions,
                #    hidden_size=impact_hidden_size,
                #    num_recurrent_layers=num_recurrent_layers,
                #    rnn_type=rnn_type,
                #    resnet_baseplanes=32,
                #    backbone=backbone,
                #    normalize_visual_inputs="rgb" in obs_space.spaces,
                #    force_blind_policy=False,
                #    policy_config=None,
                #)

                pretr_params = torch.load(pretrained_paper_encoder_weights, map_location="cpu")
                #prefix = "actor_critic.net."
                #pretr_state = {  # type: ignore
                #    k[len(prefix) :]: v
                #    for k, v in pretr_params["state_dict"].items()
                #    if k[: len(prefix)] == prefix 
                #}
                #self.obs_encoder.load_state_dict(pretr_state)
                #self.obs_encoder.load_state_dict(torch.load(pretrained_paper_encoder_weights))

                self.actor_critic.load_state_dict(
                    {  # type: ignore
                        k[len("actor_critic.") :]: v
                        for k, v in pretr_params["state_dict"].items()
                    }
                )

        else:
            self.obs_encoder = E3BBaselineNet(
                observation_spaces,
                obs_resized_dim,
                num_concat_obs,
                action_space,
                impact_hidden_size,
                #pretrained_visual_encoder,
                visual_encoder_type="resnet",
                state_encoder_type="custom",
                #state_encoder_type="rnn",
                use_rnn=use_rnn
            )
        
        if inv_model is not None:
            self.inv_model = inv_model
            self.fwd_model = fwd_model
            #if not pretrained_encoder and pretrained_visual_encoder:
            #    self.optimizer = optim.Adam(
            #        [{'params': self.inv_model.parameters()},
            #        {'params': self.obs_encoder.state_encoder.parameters()}
            #        ],
            #        lr=lr, eps=eps)
            #else:
            self.optimizer = optim.Adam(
                [
                    {'params': self.obs_encoder.parameters()},
                    {'params': self.inv_model.parameters()},
                    #{'params': self.fwd_model.parameters()},
                ],
                lr=lr, eps=eps
            )
            #self.optimizer = optim.AdamW(
            #    [
            #        {'params': self.obs_encoder.parameters()},
            #        {'params': self.inv_model.parameters()},
            #        #{'params': self.fwd_model.parameters()},
            #    ],
            #    lr=1e-4,
            #)

        self.use_normalized_advantage = use_normalized_advantage
        self.device = device
        self.use_ddp = use_ddp

    def forward(self, *x):
        raise NotImplementedError

    # Active Neural Slam
    def losses(self, features, actions_batch, T, N, features_dict=None):
        curr_states = features.view(T, N, -1)[:-1].view(
            (T - 1) * N, -1)
        next_states = features.view(T, N, -1)[1:].view(
            (T - 1) * N, -1)
        acts = actions_batch.view(T, N, -1)[:-1]
        if self.fwd_model is not None:
            acts_one_hot = torch.zeros(T - 1, N,
                                    self.fwd_model.n_actions).to(
                self.device)
            acts_one_hot.scatter_(2, acts, 1)
            acts_one_hot = acts_one_hot.view((T - 1) * N, -1)
        acts = acts.view(-1)

        if self.fwd_model is not None:
            # Forward prediction loss
            pred_next_states = self.fwd_model(curr_states.detach(),
                                            acts_one_hot)
            fwd_loss = 0.5 * F.mse_loss(pred_next_states,
                                        next_states.detach())
            #fwd_loss = F.mse_loss(pred_next_states,
            #                            next_states.detach())
            if features_dict is not None:
                features_dict['fwd_model'] = (pred_next_states.detach(), next_states.detach())
        # Inverse prediction loss
        pred_acts = self.inv_model(curr_states, next_states)
        #pred_acts = F.log_softmax(self.inv_model(curr_states, next_states), dim=-1)
        inv_loss = F.cross_entropy(pred_acts, acts.long())
        #inv_loss = F.nll_loss(pred_acts, acts.long())
        if features_dict is not None:
            features_dict['inv_model'] = (pred_acts.detach(), acts.long().detach())

        if self.fwd_model is not None:
            #tot_loss = self.curiosity_beta * fwd_loss + (1 - self.curiosity_beta) * inv_loss
            tot_loss = fwd_loss + inv_loss
            if features_dict is not None:
                return tot_loss, fwd_loss, inv_loss, features_dict
            else:
                return tot_loss, fwd_loss, inv_loss
        else:
            if features_dict is not None:
                return inv_loss, features_dict
            else:
                return inv_loss

    def update(self, rollouts, ans_cfg, flag):
        tot_loss_epoch = 0
        fwd_loss_epoch = 0
        inv_loss_epoch = 0

        data_generator = rollouts.recurrent_generator_curiosity(
            self.num_mini_batch
        )
        env = 0

        for _ in range(self.update_epochs):
            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    masks_batch,
                    adv_targ,
                    T,
                    N
                ) = sample

                self.optimizer.zero_grad()

                if flag:
                    features, rnn_hidden_states, features_dict = self.obs_encoder(obs_batch, recurrent_hidden_states_batch,
                                                               prev_actions_batch, masks_batch, flag
                                                               )
                else:
                    features, rnn_hidden_states = self.obs_encoder(obs_batch, recurrent_hidden_states_batch,
                                                               prev_actions_batch, masks_batch, flag
                                                               )

                if self.fwd_model is not None:
                    if flag:
                        tot_loss, fwd_loss, inv_loss, features_dict = self.losses(features, actions_batch, T, N, features_dict)
                    else:    
                        tot_loss, fwd_loss, inv_loss = self.losses(features, actions_batch, T, N)
                else:
                    if flag:
                        tot_loss, features_dict = self.losses(features, actions_batch, T, N, features_dict)
                    else:
                        tot_loss = self.losses(features, actions_batch, T, N)

                self.before_backward(tot_loss)
                tot_loss.backward()
                self.after_backward(tot_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                tot_loss_epoch += tot_loss.item()
                if self.fwd_model is not None:
                    fwd_loss_epoch += fwd_loss.item()
                    inv_loss_epoch += inv_loss.item()
                env += 1

        if flag:
            directory = "features/" + type(self.obs_encoder).__name__ + '/'
            if not os.path.exists(directory):
                os.mkdir(directory)

            full_path = directory + 'features_dict_update' + str(self._counter) + '.pt'
            torch.save(features_dict, full_path)
            self._counter += 25

        num_updates = self.update_epochs * self.num_mini_batch
        
        tot_loss_epoch /= num_updates
        if self.fwd_model is not None:
            fwd_loss_epoch /= num_updates
            inv_loss_epoch /= num_updates
            return tot_loss_epoch, fwd_loss_epoch, inv_loss_epoch
        else:
            return tot_loss_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.inv_model.parameters(), self.max_grad_norm
        )
        if self.fwd_model is not None:
            nn.utils.clip_grad_norm_(
                self.fwd_model.parameters(), self.max_grad_norm
            )
        nn.utils.clip_grad_norm_(
            self.obs_encoder.parameters(), self.max_grad_norm
        )

    def after_step(self):
        pass

    def to_ddp(self, device_ids, output_device):
        if self.use_ddp:
            self.pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
            #self.pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
            if self.inv_model is not None:
                self.inv_model = nn.parallel.DistributedDataParallel(self.inv_model, device_ids=device_ids,
                                                                     output_device=output_device,
                                                                     process_group=self.pg1)
            #if self.fwd_model is not None:
                #self.fwd_model.to_ddp(device_ids, output_device)
            #if self.paper_encoder:
            #    self.obs_encoder = nn.parallel.DistributedDataParallel(self.obs_encoder, device_ids=device_ids,
            #                                                         output_device=output_device,
            #                                                         process_group=self.pg2)
            #else:
            #    self.obs_encoder.to_ddp(device_ids, output_device)
