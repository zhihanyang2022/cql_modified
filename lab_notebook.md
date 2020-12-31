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

## Random and expert baseline for each task



