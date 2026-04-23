# General-Sum Car Game - CLI

```
python cargame_g.py [OPTIONS]
```

--iterations  Number of planning iterations (default: 8000)
--grid-size   Size of the grid (default: 5)

## Module Layout

- `cargame_g.py`: thin CLI entrypoint and orchestration
- `config.py`: hyperparameters and constants
- `dqn.py`: DQN model, replay buffer, state encoding
- `environment.py`: `CarGame` transition and reward logic
- `game_theory.py`: Nash equilibrium and fast Nash-value helpers
- `trainer.py`: Nash-Q training loop
- `policy.py`: policy extraction and rollout
- `visualization.py`: rollout plots and loss plots
- `io_utils.py`: output paths and weight export
