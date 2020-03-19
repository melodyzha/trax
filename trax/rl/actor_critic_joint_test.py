# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for RL training."""

from absl.testing import absltest

from trax import layers as tl
from trax import lr_schedules
from trax import optimizers as opt
from trax.rl import actor_critic_joint
from trax.rl import task as rl_task


class ActorCriticJointTest(absltest.TestCase):

  def test_jointawrtrainer_cartpole(self):
    """Test-runs joint AWR on cartpole."""
    task = rl_task.RLTask('CartPole-v0', initial_trajectories=1000,
                          max_steps=200)
    shared_model = lambda mode: tl.Serial(tl.Dense(64), tl.Relu())
    policy_top = lambda mode: tl.Serial(tl.Dense(2), tl.LogSoftmax())
    value_top = lambda mode: tl.Dense(1)
    lr = lambda h: lr_schedules.MultifactorSchedule(  # pylint: disable=g-long-lambda
        h, constant=1e-2, warmup_steps=100, factors='constant * linear_warmup')
    trainer = actor_critic_joint.AWRJointTrainer(
        task,
        shared_model=shared_model,
        policy_top=policy_top,
        value_top=value_top,
        optimizer=opt.Adam,
        lr_schedule=lr,
        batch_size=32,
        train_steps_per_epoch=1000,
        collect_per_epoch=10)
    trainer.run(1)
    self.assertEqual(1, trainer.current_epoch)
    self.assertGreater(trainer.avg_returns[-1], 150.0)


if __name__ == '__main__':
  absltest.main()
