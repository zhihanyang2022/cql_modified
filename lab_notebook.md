# Log of experiments

[TOC]

## Count number of gradient steps

Note crucially that: the `rlkit` package used by `CQL` has been modified compared to the original `rlkit` implementation. The following discussion refers to the first one.

It seems that the number of gradient steps performed is determined by 3 parameters to the `TorchBatchRLAlgorithm` class (see `cql_mujoco_count_gradient_steps.py` line 98), which are:

- `num_epochs`: number of outermost loops; alternate between (1) evaluation and (2) **training**
- `num_train_loops_per_epoch`: number of iterations by which we alternate between (1) collecting new paths (ignored by CQL) and (2) **training on buffer**
- `num_trains_per_train_loop`: number of times that we (1) sample from the buffer and then (2) perform a **gradient step**

In summary, the total number of gradient steps performed should be the three numbers multiplied together. However, since `num_train_loops_per_epoch=1` by default, the total number of gradient steps performed should be 

```
num_epochs * num_trains_per_train_loop
```

Now, we consider proving this understanding empirically but counting the number of gradient steps actually performed during a toy run. For this toy run, we have chosen the following parameters:

```python
# line 131, cql_mujoco_count_gradient_steps.py

algorithm_kwargs=dict(
	num_epochs=17,  # first determinant
	num_eval_steps_per_epoch=100,
	num_trains_per_train_loop=19,  # second determinant
	num_expl_steps_per_train_loop=100,
	min_num_steps_before_training=0,
	max_path_length=100,
	batch_size=10,
),

# therefore, we expect the number of gradient steps to be 17 * 19 = 323
```

`cql_mujoco_count_gradient_steps.py` has added an additional line of code after `algorithm.train()`: `print(trainer._n_train_steps_total)`. Running this code indeed gives 323.

Additional parameters in Pycharm config:

```
# py script variables
--env=hopper-medium-v0
--policy_lr=3e-5
--seed=10
--lagrange_thresh=10.0
--min_q_weight=1
--min_q_version=3

# environment variables
PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/home/zhihanyang/.mujoco/mujoco200/bin
```

## SAC for gym mujoco

![image-20210101151600791](https://i.loli.net/2021/01/01/payTe3bZchMfn6N.png)

![image-20210101151641611](https://i.loli.net/2021/01/01/VMAncRfyHvljiIu.png)

Performance of SAC learned temperature:

-   Hopper-v2: around 3000
-   Walker2d-v2: approaching 6000
-   HalfCheetah-v2: approaching 15000

## Random policy baseline (for hopper only)

This section discusses how the random-policy baseline (in terms of average returns) is computed using the existing CQL code-base. First, to do this, we have created a separate script called `examples/cql_mujoco_hopper_random_policy.py`.

The arguments to the `BatchRLAlgorithm` class in `rlkit/core/batch_rl_algorithm.py ` are

- `batch_rl = True` (line 195 of `examples/cql_mujoco_hopper_random_policy.py`)
- `eval_both = True` (line 101 `examples/cql_mujoco_hopper_random_policy.py`.)
- `self.q_learning_alg = False` (default)

```python
algorithm_kwargs=dict(
            num_epochs=1,
            num_eval_steps_per_epoch=10000,
            num_trains_per_train_loop=0,  # set to zero to avoid training
            num_expl_steps_per_train_loop=0,  # set to zero to avoid training
            min_num_steps_before_training=0,  # set to zero to avoid training
            max_path_length=1000,
            batch_size=256,  # set to zero to avoid training
),
```

These settings simplifies the `_train ` method . Code that are not ran due to these parameter settings are commented out below. Essentially, these settings change the training code into evaluation-only code: the code runs for one epoch only, completing however many paths it can, depending on stochasticity of the policy, `max_path_length` and `max_eval_steps_per_epoch`. As a word of caution, make sure to check that the number of gradient steps completed is zero.

```python
   def _train(self):
        # if self.min_num_steps_before_training > 0 and not self.batch_rl:
        #     init_expl_paths = self.expl_data_collector.collect_new_paths(
        #        self.max_path_length,
        #        self.min_num_steps_before_training,
        #        discard_incomplete_paths=False,
        #    )
        #    self.replay_buffer.add_paths(init_expl_paths)
        #    self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            # if self.q_learning_alg:
            #    policy_fn = self.policy_fn
            #    if self.trainer.discrete:
            #        policy_fn = self.policy_fn_discrete
            #    self.eval_data_collector.collect_new_paths(
            #        policy_fn,
            #        self.max_path_length,
            #        self.num_eval_steps_per_epoch,
            #        discard_incomplete_paths=True
            #    )
            else:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                # if not self.batch_rl:
                #    # Sample new paths only if not doing batch rl
                #    new_expl_paths = self.expl_data_collector.collect_new_paths(
                #        self.max_path_length,
                #        self.num_expl_steps_per_train_loop,
                #        discard_incomplete_paths=False,
                #    )
                #    gt.stamp('exploration sampling', unique=False)
				#
                #    self.replay_buffer.add_paths(new_expl_paths)
                #    gt.stamp('data storing', unique=False)
                elif self.eval_both:
                    # Now evaluate the policy here:
                    policy_fn = self.policy_fn
                    # if self.trainer.discrete:
                    #     policy_fn = self.policy_fn_discrete
                    # new_expl_paths = self.expl_data_collector.collect_new_paths(
                    #     policy_fn,
                    #     self.max_path_length,
                    #     self.num_eval_steps_per_epoch,
                    #     discard_incomplete_paths=True,
                    #)

                    gt.stamp('policy fn evaluation')

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    # train_data = self.replay_buffer.random_batch(
                    #     self.batch_size)
                    # self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
```

