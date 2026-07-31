what I changed to this repo this past week:

# New folder
Newly added folder: /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output

no code change in scripts, just got output from training finetuning, output paths:
1. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Labram/debug_output
2. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/EEGPT/debug_output
3. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/CSBrain/debug_output
4. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Codebrain/debug_output
5. //Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/CBraMod
6. /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM/Biot

Newly added document named 'terminal command' under /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output, it only has 6 terminal command used to run training finetuning and get output. 

How I got the training done: the idea was not changing any exisiting codes in /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output/EEG-FM, but used external connector runner scripts to connect TUAB.pkl subset to the traiing scripts.

EEGMamba do not have output because AIstation environment conflict. 

Note: needs debugging, will further discuss what to debug.

# New lines of code added
To connect from /Noise-Robustness-Evaluation-Protocol/Benchmark to finetuning scripts in /Noise-Robustness-Evaluation-Protocol/EEG-FM,  a few lines of new codes were added to training scripts in /Noise-Robustness-Evaluation-Protocol/EEG-FM.

You will know which codes were newly added with #newly added codes.

# Deleted files
Old README.md from /Noise-Robustness-Evaluation-Protocol/Benchmark was deleted to avoid confusion.

/Noise-Robustness-Evaluation-Protocol-demo/Benchmark/Evaluation was deleted to avoid confusion, there's no use for it if we will have a unified ouput folder such as /Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output

# yet another new folder produced
Noise-Robustness-Evaluation-Protocol/Benchmark/Outputs/save_finetune_checkpoints was produced from loading TUAB H5 subsets and train finetuning from /Noise-Robustness-Evaluation-Protocol/EEG-FM. The outputs were not unified, but it proves that our benchmark loader scripts work, and it works well with training scripts from /Noise-Robustness-Evaluation-Protocol/EEG-FM.

# PDF file added
This pdf shows a table of default training hyperparameters for each model, manually checked from training scripts from /Noise-Robustness-Evaluation-Protocol/EEG-FM.

# what mainly changed in this repo from last time
1. /Noise-Robustness-Evaluation-Protocol/Benchmark/Config: I checked though all .yaml files again, so there are code change (just a bit) after our last discussion.
2. I completely rewrote the code in /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining. Only 6 scripts, the only function is the connector loads YAML as
defaults through the author script and only appends explicit CLI overrides. You can ignore /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining/Tests, it was left in the folder to remind me that labram 

# This folder will not be used but important to keep
/Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining/Tests
There is a README.md to explain the reason why the script in this folder exist.

# Notes for later discussion:
1. Please look through /Noise-Robustness-Evaluation-Protocol/Benchmark/LoaderTraining. I selected and added in the default params from /Noise-Robustness-Evaluation-Protocol/EEG-FM training scripts. The training runs also work. But I might still miss something. 
2. /nicoletye/workspace/Noise-Robustness-Evaluation-Protocol/debug-unified-logging-output was created after running a short real training with 2 epochs. I inspected the outputs, the details need to be discussed face-to-face. 
3. I think the names of folders and scripts from  /Noise-Robustness-Evaluation-Protocol/Benchmark are not well-written. We shoulld discuss and rename them to avoid confusion. 
