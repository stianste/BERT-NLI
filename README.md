# BERT-NLI
A PyTorch implementation using the BERT network from Google to solve the task of NLI.

## How to get started
Install virtualenv and create a virtual environment by running
`pip3 install virtualenv`
`mkdir venv`
`virtualenv --python=/usr/bin/python3.6 venv/`

Donwload the example data by running
`cd data/`
chmod +x download_example_data.sh`
`./download_example_data.sh`

Next you can run the CoLA example by going into the `examples` folder and running
`chmod +x run_CoLA.sh`
`./run_CoLA.sh`
