# Scripts

The published project scripts are grouped by workflow:

- `setup/`: dependency and pretrained-asset setup;
- `training/`: end-to-end training and validation pipelines;
- `preparation/`: experiment manifests, configurations, references, and masks;

Run Python entry points as modules from the repository root. For example:

```bash
python -m scripts.training.run_atlas_validation --help
python -m scripts.training.run_method_baselines --help
python -m scripts.training.run_segment_stress_validation --help
```
