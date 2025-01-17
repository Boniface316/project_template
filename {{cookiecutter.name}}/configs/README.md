# Configs files

These files are used to tweak the parameters defined in the schema. Thefore, instead of adjusting parameters on the code, we can explcitly define the values. 

By default, you need to define job kind, inputs and targets. An example of YAML file is provided below.

```yaml
job:
  KIND: TrainingJob
  inputs:
    KIND: ParquetReader
    path: data/inputs_train.parquet
  targets:
    KIND: ParquetReader
    path: data/targets_train.parquet
```

## Mandatory fields
Dictaed by each jobs. See the fields that needs to be filled. Schema will show the required fields.

### job
#### KIND
This has to be the kind of job you want to execute. The list of jobs are defined in 

### Using extras
uv run mlops-project configs/config.yaml --extras '{"job": {"running_msg": "from terminal"}}'