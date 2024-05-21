# Generate Workload
Currently, two types of workload are supported.
- `dummy`: A synthetic input, output pair with a given length range.
- `hf_{dataset_name}`: A input, output pair from huggingface dataset. Currently supported type are:
    - `hf_cnn_daily_mail`
    - `hf_dolly`
    - `hf_alpaca`
    - `hf_sharegpt`
If you want to add other workload type, then add other class inherited from `RequestDataset` in `src/workload`.

> [!NOTE]
> Currently, for using huggingface dataset, you must add other class inherited from `RequestDataset` in `src/workload`. Also, the `type` field in `workload_config.yaml` must be `hf_{dataset_name}`.


## Dummy Workload Format
There are three field in `dataset_config` filed of workload_config for Dummy Workload.
- `vocab_size`: the vocab size of served model.
- (Optional)`system_prompt_config`: The list of system prompt with fixed length. Each system prompt have `name` and fixed `size`.
- `dataset`: The list of (input, output, system_prompt) pair.
    - `input`: The list of input sampler configuration. Each input sampler have distribution `type`(uniform/normal) and input length range(`min`, `max`). Each sampler has `weight`, that is the weight of the sampler. The default value is 1.
    - `output`: The list of output sampler configuration. Each output sampler have distribution `type`(uniform/normal) and output length range(`min`, `max`). Each sampler has `weight`, that is the weight of the sampler. The default value is 1.
    - (Optional)`system_prompt`: The list of system prompt sampler configuration. Select system prompts by `name` from `system_prompt_config` and choose the one with `weight`. The default `weight` is 1.
    - `weight`: The weight of the (input, output, system_prompt) pair. The default value is 1.

#### Example of `workload_config.yaml` for Dummy Workload:
```yaml
# dummy workload_config.yaml
type: dummy
dataset_config:
  vocab_size: 128000 # must be same as the vocab size of the served model.
  system_prompt_config:
  - name: "1"
    size: 10
  - name: "2"
    size: 20
  dataset:
  - input:
    - type: uniform
      min: 100
      max: 150
      weight: 1
    output:
    - type: uniform
      min: 100
      max: 300
      weight: 1
    system_prompt:
    - name: "1"
      weight: 1
    - name: "2"
      weight: 2
    weight: 1
```

### Huggingface Dataset Workload Format
There are five field in `dataset_config` filed of workload_config for Huggingface Dataset Workload.

- `max_length`: The maximum length of input and output. If there are input or output longer than this value, then it will be not used.
- `min_length`: The minimum length of input and output. If there are input or output shorter than this value, then it will be not used.
- `path_or_name`: The repo id of the huggingface dataset or the local path of the dataset.
- `format`: The format of the dataset.
- `split`: The split of the dataset.

> [!NOTE]
> **Dataset Field: `path_or_name`, `format`, `split`**
> These field are used for loading the dataset from huggingface dataset with [`datasets.load_dataset`](https://huggingface.co/docs/datasets/loading).
