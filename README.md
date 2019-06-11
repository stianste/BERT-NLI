# BERT-NLI
A PyTorch implementation using BERT from Google to solve the task of NLI.
This repo is the code used to obtain a Masters in Computer Science at NTNU. 

To achieve this, the [PyTorch implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT) has been used.
This repo is not officially maintaned by Google, but the official Tensorflow repo
links to this one as a good PyTorch implementation which achieves the same results
as the original implementation. 

## How to get started
Install virtualenv and create a virtual environment by running
```shell
pip3 install virtualenv
mkdir venv
virtualenv --python=/usr/bin/python3.6 venv/
```
This is not necessary on the remote server, as a python environment is set up
for each slurm job.

### IDUN
As the system requires quite heavy computational power, the experiments have
been run on [IDUN](https://www.hpc.ntnu.no/display/hpc/Getting+Started+on+Idun).
To login on IDUN, run 

`ssh -l stianste idun-login3.hpc.ntnu.no`

where `stianste` is the username, and then 

`sbatch run_job.slurm`

to run the job contained in `run_job.slurm`. To watch the job run 

`watch -n 1 squeue -u stianste`

### Running the TOEFL11 example
In order to run the TOEFL11 example on the remote host, you need to transfer the exisiting
directory over ssh. Assuming that the data is avaliable in the `data` folder, and
depending on where on the server you want to put it:

```shell
scp -r data/NLI-shared-task-2017 stianste@idun-login3.hpc.ntnu.no:/lustre1/work/stianste/BERT-NLI/ data
```

After that, one can run BERT on the TOEFL11-data set by running `sbatch slurm_jobs/toefl.slurm`.
The output of the job will be located in `outputs/toefl.out`. In order to watch
the output while the job is running, run `tail -f outputs/toefl.out`.
The same goes for `slurm_jobs/reddit.slurm`, which runs BERT using cross-validation
over the Reddit-L2 data set.

## Important files
Most of the code neccessary to run BERT is located in the `run_BERT_NLI.py` file.
To see an example of what arguments this script requires and how to run it,
see the `toefl.slurm` script. Furthermore, the `data_processors.py` file
is used to read in test and training data. As long as the custom DataProcessor
class implements the basic methods required, it can be loaded in `run_BERT_NLI.py`
and used for training and testing BERT.

The different ensembles can be found in `simple_ensemble.py` – which trains the ensemble
on the TOEFL11 data set – and the `reddit_ensemble.py` script which trains an ensemble
on the Reddit-L2 data. If BERT is to be used in combination with the ensemble,
the test and training outputs of BERT must be located in the same folder
as the predictions of the base classifiers.