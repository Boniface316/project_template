# Configs files

These files are used to tweak the parameters defined in the schema. Thefore, instead of adjusting parameters on the code, we can explcitly define the values. 

[example.yaml](example.yaml) can be used as an example to see how the arguments can be supplied.

## Understanding the schema

Run `uv run [package name] configs/example.yaml > schema.json` to get the schema of the input. It would appear in the form below. Keys and values that goes in the in the sub keys are removed for simplicity.


```json
{
    "$defs": {
        "AlertsService": {
            ...
        },
        "BuiltinSaver": {
            ...
        },
        "CustomSaver": {
            ...
        },
        "ExampleMetric": {
            ...
        },
        "ExampleModel": {
            ...
        },
        "ExampleReader": {
            ...
        },
        "ExampleSearcher": {
            ...
        },
        "ExampleSigner": {
            ...
        },
        "ExampleSplitter": {
            ...
        },
        "LoggerService": {
            ...
        },
        "MlflowRegister": {
            ...
        },
        "MlflowService": {
            ...
        },
        "RunConfig": {
           ...
        },
        "TrainingJob": {
            ...
        },
        "TuningJob": {
            ...
        }
    },
    "additionalProperties": false,
    "description": "Main settings of the application.\n\nParameters:\n    job (jobs.JobKind): job to run.",
    "properties": {
        "job": {
            "discriminator": {
                "mapping": {
                    "TrainingJob": "#/$defs/TrainingJob",
                    "TuningJob": "#/$defs/TuningJob"
                },
                "propertyName": "KIND"
            },
            "oneOf": [
                {
                    "$ref": "#/$defs/TrainingJob"
                },
                {
                    "$ref": "#/$defs/TuningJob"
                }
            ],
            "title": "Job"
        }
    },
    "required": [
        "job"
    ],
    "title": "MainSettings",
    "type": "object"
}
```json

In the YAML file, you need to define the `KIND` of the job. The available variables are defined in the properties as follow:

```json

"properties": {
        "job": {
            "discriminator": {
                "mapping": {
                    "TrainingJob": "#/$defs/TrainingJob",
                    "TuningJob": "#/$defs/TuningJob"
                },
                "propertyName": "KIND"
            },
            "oneOf": [
                {
                    "$ref": "#/$defs/TrainingJob"
                },
                {
                    "$ref": "#/$defs/TuningJob"
                }
            ],
            "title": "Job"
        }
    },
```

In this example, you have a choice between `TrainingJob` and `TuningJob`. Then expand the job key under `$defs`. Below is an example of `TrainingJob` schema:

```json
"TrainingJob": {
    ...
  "properties": {
    ...
  },
  "required": [
    "inputs",
    "targets"
  ],
  "title": "TrainingJob",
  "type": "object"
},

```

The items under `required` key will inform the required arguments to run this job. They are `inputs` and `targets`. If we goto to the `inputs` key, we see the following:

```json

 "inputs": {
  "discriminator": {
      "mapping": {
          "ExampleReader": "#/$defs/ExampleReader"
      },
      "propertyName": "KIND"
  },
  "oneOf": [
      {
          "$ref": "#/$defs/ExampleReader"
      }
  ],
  "title": "Inputs"
}
```

In the mapping, we can see that `ExampleReader` is mapped to input. If we want to use that, we need to find the argument it requires. Therefore, we need look at `ExampleReader` key and see the following:

```json
"ExampleReader": {
    "additionalProperties": false,
    "description": "Read a dataframe from an example dataset.",
    "properties": {
        ...
    },
    "required": [
        "path"
    ],
    "title": "ExampleReader",
    "type": "object"
},
```

Here we can see that it requires `path` argument. We can do the same for `targets` to find what we know. Now we have all the keys and variables to create a config file. Lets put them in YAML file as follows:

```yaml
job:
  KIND: TrainingJob
  inputs:
    KIND: ExampleReader
    path: data/input.csv
  targets:
    KIND: ExampleReader
    path: data/targets.csv
```
## Changing variables

First we need to find an argument that you want to change. This should be under the job that you want to submit. Assume that we want to change the train/test split. Lets examine the `splitter` key:

```json

 "splitter": {
    "default": {
        "KIND": "ExampleSplitter",
        "shuffle": false,
        "test_size": 0.2,
        "random_state": 42
    },
    "discriminator": {
        "mapping": {
            "ExampleSplitter": "#/$defs/ExampleSplitter"
        },
        "propertyName": "KIND"
    },
    "oneOf": [
        {
            "$ref": "#/$defs/ExampleSplitter"
        }
    ],
    "title": "Splitter"
},
```

Here we can see that it has `ExampleSplitter` assigned to it. The `ExampleSplitter schema is:

```json
"ExampleSplitter": {
    "additionalProperties": false,
    "properties": {
        "KIND": {
            "const": "ExampleSplitter",
            "default": "ExampleSplitter",
            "title": "Kind",
            "type": "string"
        },
        "shuffle": {
            "default": false,
            "title": "Shuffle",
            "type": "boolean"
        },
        "test_size": {
            "anyOf": [
                {
                    "type": "integer"
                },
                {
                    "type": "number"
                }
            ],
            "default": 0.2,
            "title": "Test Size"
        },
        "random_state": {
            "default": 42,
            "title": "Random State",
            "type": "integer"
        }
    },
    "title": "ExampleSplitter",
    "type": "object"
}
```

Unlike the previous example, this one does not have any `required` key. Each sub-key under properties are variables that can be defined using configs file. Assume that we want to change `test_size`, here is how it would reflect in the YAML file:

```YAML
job:
  KIND: TrainingJob
  inputs:
    KIND: ExampleReader
    path: data/input.csv
  targets:
    KIND: ExampleReader
    path: data/targets.csv
  splitter:
    KIND: ExampleSplitter
    test_size: 0.5
```

