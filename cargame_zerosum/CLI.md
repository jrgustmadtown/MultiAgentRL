# Zero-Sum Car Game - CLI

```
python cargame_z.py [OPTIONS]
```

--iterations  Number of planning iterations (default: 8000)
--grid-size   Size of the grid (default: 5)

## Module Layout

- `cargame_z.py`: thin CLI entrypoint and orchestration
- `config.py`: hyperparameters and constants
- `dqn.py`: DQN model, replay buffer, state encoding
- `environment.py`: `CarGame` transition and reward logic
- `trainer.py`: minimax DQN training loop
- `policy.py`: policy extraction and rollout
- `visualization.py`: rollout plots and loss plots
- `io_utils.py`: output paths and weight export
