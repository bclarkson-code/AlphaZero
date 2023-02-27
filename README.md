# AlphaZero

A simple implementation of the AlphaZero algorithm for the game of Draughts (called checkers in the 
USA). Note, the code can be described as "technically good enough for a uni project" and is by no
means production ready.

The project consists of a network, described in the `network.py` file, the training algorithm in 
`alphazero.py` and an implementation of Noughts and Crosses (TicTacToe) and Draughts (Checkers).
For speed, Draughts was implemented in Cython.

During execution, the core of the algorithm worked and achieved statistically significant learning after 2 
days of computation. That said, while this algorithm was run for 1 training loop, DeepMind ran 
theirs for 700,000 so their AI is probably slightly better than this one.

## Installation

Conda is used to manage the environment for this project. This process has only been tested on my 
personal macbook.


```
conda env create -n alphazero -f tf-metal-arm64.yaml
conda activate alphazero
```

## Run

The training can be started as follows:
```
python alphazero.py
```