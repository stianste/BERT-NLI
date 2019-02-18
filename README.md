# BERT-NLI
A PyTorch implementation using the BERT network from Google to solve the task of NLI.

To achieve this, the [PyTorch implementation of BERT](https://github.com/huggingface/pytorch-pretrained-BERT) has been used.
This repo is not officially maintaned by Google, but the official Tensor Flow repo
links to this one as a good PyTorch implementation which achieves the same results
as the original implementation. 

## How to get started
Install virtualenv and create a virtual environment by running
```shell
pip3 install virtualenv
mkdir venv
virtualenv --python=/usr/bin/python3.6 venv/
```

Donwload the example data by running
```shell
cd data/
chmod +x download_example_data.sh
./download_example_data.sh
```

Next you can run the CoLA example by going into the `examples` folder and running
```shell
../examples/
chmod +x run_CoLA.sh
./run_CoLA.sh
```

### IDUN
To login on [IDUN](https://www.hpc.ntnu.no/display/hpc/Getting+Started+on+Idun) run 

`ssh -l stianste idun-login3.hpc.ntnu.no`

and then 

`sbatch run_job.slurm`

to run the job. To watch the job run 

`watch -n 1 squeue -u stianste`

### Memory issues
On most machines, this will not run out of the box due to memory issues.
For this reason, I have been granted access to the NTNU-Luke01 Nvidia Tesla
P100 GPU. In order to access the machine, run

```shell
ssh username@login.stud.ntnu.no
ssh luke01.idi.ntnu.no
```

And enter your NTNU password when prompted.

## Running the TOEFL11 example
In order to run the TOEFL11 example on the remote host, you need to transfer the exisiting
directory over ssh. Assuming that the data is avaliable in the `data` folder:

```shell
scp -r data/NLI-shared-task-2017 stianste@luke01.idi.ntnu.no:/work/lhome/stianste/projects/BERT-NLI/data
```

After than, one can just give permission to the `run_NLI.sh` script and run it:
```shell
chmod +x run_NLI.sh
./run_NLI.sh
```