## Training
## dependency structure:
Training/train_<model>.py
    │
    ├── uses Config/<model>.yaml
    │
    ├── imports Training/training_common.py
    │
    ├── imports the author model repository from paths.model_repo
    │
    ├── calls Loader/loader_<model>.py through training_common.py
    │
    └── calls StudyCase/Channel/<Model>_23ch_vs_16ch.py when a channel study is selected

## Executable flow:
When a training script starts, the flow is as follows:

main()
    ↓
parse_arguments()
    ↓
load_config()
    ↓
run_experiments()
    ↓
for each seed:
    run_seed()


## Inside each seed:

set random seed
    ↓
choose CPU or GPU
    ↓
create output folder
    ↓
save visible config
    ↓
build train/validation/test loaders
    ↓
build optional study-case transform
    ↓
build model from author repository
    ↓
load pretrained checkpoint
    ↓
configure fine-tuning strategy
    ↓
build optimizer and scheduler
    ↓
train for each epoch
    ↓
select best validation epoch
    ↓
reload best model
    ↓
evaluate test split once
    ↓
save seed result

After all seeds finish:

summarize_seed_results()
    ↓
save all_seed_results.json
save summary.json