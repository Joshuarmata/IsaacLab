# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Student network for the camera policy.

The picture and the pose vector stay separate so the rollout buffer does not
store a copy of the pose on every pixel. The convolution reads the overhead RGB and depth.
Its features are then joined with the joint angles, the base position in the camera frame,
and the saved strap depth.

The privileged critic is a flat vector stored under one dict key. This builder flattens that
dict so the stock value MLP still applies.
"""

from __future__ import annotations

import torch
from rl_games.algos_torch import network_builder
from rl_games.algos_torch.model_builder import register_network
from torch import nn


class HandleGrabCameraBuilder(network_builder.A2CBuilder):
    """Actor-critic builder for the handle-grab camera policy."""

    def build(self, name, **kwargs):
        input_shape = kwargs["input_shape"]
        if isinstance(input_shape, dict) and "image" in input_shape:
            proprio_size = int(input_shape["proprio"][0])
            kwargs["input_shape"] = input_shape["image"]
            return HandleGrabCameraBuilder.StudentNetwork(self.params, proprio_size=proprio_size, **kwargs)
        if isinstance(input_shape, dict):
            flat = 0
            for shape in input_shape.values():
                size = 1
                for dim in shape:
                    size *= int(dim)
                flat += size
            kwargs["input_shape"] = (flat,)
            return HandleGrabCameraBuilder.FlatNetwork(self.params, **kwargs)
        return network_builder.A2CBuilder.Network(self.params, **kwargs)

    class FlatNetwork(network_builder.A2CBuilder.Network):
        """Value network. The critic observation arrives as a one-entry dict."""

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            if isinstance(obs, dict):
                flat = torch.cat([value.reshape(value.shape[0], -1) for value in obs.values()], dim=-1)
                obs_dict = {**obs_dict, "obs": flat}
            return super().forward(obs_dict)

    class StudentNetwork(network_builder.A2CBuilder.Network):
        """CNN on the picture, then the pose vector joined to the flattened features."""

        def __init__(self, params, proprio_size: int, **kwargs):
            image_shape = kwargs["input_shape"]
            super().__init__(params, **kwargs)
            self.proprio_size = proprio_size
            dummy = torch.zeros((1, *image_shape))
            if self.permute_input:
                dummy = dummy.permute(0, 3, 1, 2)
            cnn_out = int(self.actor_cnn(dummy).flatten(1).shape[1])
            self.actor_mlp = self._build_mlp(
                input_size=cnn_out + proprio_size,
                units=self.units,
                activation=self.activation,
                norm_func_name=self.normalization,
                dense_func=nn.Linear,
                d2rl=self.is_d2rl,
                norm_only_first_layer=self.norm_only_first_layer,
            )
            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.actor_mlp.modules():
                if isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            image = obs["image"]
            if self.permute_input and image.dim() == 4:
                image = image.permute(0, 3, 1, 2)
            features = self.actor_cnn(image).flatten(1)
            out = self.actor_mlp(torch.cat((features, obs["proprio"]), dim=-1))
            value = self.value_act(self.value(out))
            mu = self.mu_act(self.mu(out))
            if self.fixed_sigma:
                sigma = self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(out))
            return mu, mu * 0.0 + sigma, value, states


register_network("handle_grab_camera", HandleGrabCameraBuilder)
