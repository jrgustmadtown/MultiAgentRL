# Dog Game Coop - CLI

```bash
python doggame_coop.py [OPTIONS]
```

--iterations     Training iterations (default: 8000)
--step-size      Movement step size in [0,1] space (default: 0.1)
--horizon        Rollout episode length (default: 20)
--house          Single dog house position x,y (default: 0.75,0.75)
--success-radius Terminal success radius around house (default: 0.05)
--wall           Wall segment x0,y0,x1,y1 (default: 0.5,0.0,0.5,0.75)

## Module Layout

- doggame_coop.py: thin CLI entrypoint and orchestration
- config.py: constants and hyperparameters
- dqn.py: model, replay buffer, state encoding
- environment.py: cooperative dog midpoint dynamics and rewards
- trainer.py: cooperative planning-Q training loop
- policy.py: cooperative policy extraction and rollout
- visualization.py: rollout and loss plotting
- io_utils.py: output paths and weight export

## Shared Utilities

- utilities/replay.py: shared ReplayBuffer
- utilities/paths.py: shared output directory/path helpers
- utilities/runtime.py: shared runtime helper (headless display detection)
