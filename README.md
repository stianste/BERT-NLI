# BERT-NLI
A PyTorch implementation using the BERT network from Google to solve the task of NLI.

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

### Memory issues
On most machines, this will not run out of the box due to memory issues.
For this reason, I have been granted access to the NTNU-Luke01 Nvidia Tesla
P100 GPU. In order to access the machine, run

```shell
ssh username@login.stud.ntnu.no
ssh luke01.idi.ntnu.no
```

And enter your NTNU password when prompted.