# The Ultimate Poker Odds Calculator 
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
* ```PokerOddsCalculator.py``` - The main script that runs the program
* ```PlayingCardClassifier.py``` - The script that trains the machine learning model that classifies the playing cards
* ```CardDatasetCreator.ipynb``` - A jupyter notebook that creates the training/validation dataset for the classifier

### Requirements

The following modules and their repsective versions are used in this project, 
* OpenCV 4.10.0
* PyTorch (torch) 2.5.1
* Numpy 1.26.4
* tqdm 4.67.1
* tabulate 0.9.0

### Basic Setup and Usage
1. Begin with cloning the repository, 
```
git clone https://github.com/VinVincely/PokerOddsCalculator/tree/main
```
2. Next, build the dataset necessary to train the classifier by following the steps outlined in the juypter notebook. NOTE: This is a crucial step as the performance of the classifier (which is a traditional convolutional neural network) is highly dependent of the lighting conditions that the user presents. Hence, I recommend that the user re-trains the model prior to using the calculator. 

3. Train the classifier using ``` PlayingCardClassifier.py``` as follows, 
```
python.exe .\PlayingCardClassifier.py 
```

4. Finally, run the main script that loads the camera feed and calls the trained classifier using the following call, 
```
python.exe .\PokerOddsCalculator.py
```

### Contact 

Please feel free to direct your comments/suggestion or general questions to vinoin.vincely@gmail.com 
