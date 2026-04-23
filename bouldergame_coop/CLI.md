# Boulder Cooperative Game - CLI (Scaffold)

```bash
python bouldergame.py [OPTIONS]
```

--iterations  Training iterations (default: 8000)
--step-size   Player movement step size (default: 0.1)
--horizon     Episode length (default: 30)
--hole        Hole position x,y (default: 0.8,0.8)

## Status

Step 2 scaffold is complete.

- Environment transitions: complete (Step 3)
- Reward implementation: complete (Step 4)
- Training loop: complete (Step 5)
- Visualizations: complete (Step 6)

## Module Layout

- `bouldergame.py`: thin CLI entrypoint
- `config.py`: constants and defaults
- `dqn.py`: model and state encoding
- `environment.py`: environment API scaffold
- `trainer.py`: training scaffold
- `policy.py`: policy scaffold
- `visualization.py`: visualization scaffold
- `io_utils.py`: output path and export helpers
- `SPEC.md`: agreed mechanics specification
