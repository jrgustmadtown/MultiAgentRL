# Dog Game - CLI

```
python doggame.py [OPTIONS]
```

--iterations  Training iterations (default: 8000)
--step-size   Movement step size in [0,1] space (default: 0.1)
--horizon     Episode length (default: 10)
--house1      House 1 position x,y (default: 0.25,0.25)
--house2      House 2 position x,y (default: 0.75,0.75)

## Module Layout

- `doggame.py`: thin CLI entrypoint and orchestration
- `config.py`: constants and hyperparameters
- `dqn.py`: model, replay buffer, state encoding
- `game_theory.py`: Nash equilibrium helpers
- `environment.py`: `DogGame` transitions and rewards
- `trainer.py`: Nash-Q training loop
- `policy.py`: policy extraction and rollout
- `visualization.py`: rollouts, vector field, and loss plots
- `io_utils.py`: output paths and weight export
