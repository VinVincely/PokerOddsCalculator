# The Ultimate Poker Odd Calculator 
A sophisticated poker odds calculator that uses computer vision and machine learning to detect cards and calculate winning probabilities through Monte Carlo simulation.

![Example](Images/coverImage.png)

## Features

* Real-time Card detection 
* Accurate probability calculations using Monte Carlo simulation
* Pre- and post-river odds calculation 
* Texas Hold'em poker rules applied
* Win rate, split rate and total equity computed

## Installation 

This project contains the following three files that each perform an essential task to setting up the odds calculator. 
* PokerOddsCalculator.py - The main script that runs the program
* PlayingCardClassifier.py - The script that trains the machine learning model that classifies the playing cards
* CardDatasetCreator.ipynb - A jupyter notebook that creates the training/validation dataset for the classifier

### Requirements

The following modules and their repsective versions are used in this project, 
* OpenCV 4.10.0
* PyTorch (torch) 2.5.1
* Numpy 1.26.4
* tqdm 4.67.1
* tabulate 0.9.0

### Basic Setup
```
git clone https://github.com/VinVincely/PokerOddsCalculator/tree/main
```


### Contact 

Please feel free to direct your comments/suggestion or general questions to vinoin.vincely@gmail.com 
Project Link: 