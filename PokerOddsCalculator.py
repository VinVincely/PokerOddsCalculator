# ========== Poker Hands ==========
# The objective of this program is to calculate the odds of a given poker hand. We will access the device's camera, identify the cards in play and then proceed to print out the odds of winning. 
#
# Author: Vinoin Devpaul Vincely (2025)

from collections import namedtuple
from PIL import Image 
import random, torch
import matplotlib.pyplot as plt 
import re, cv2, numpy as np
from tabulate import tabulate
import PlayingCardClassifier as cardNN
from PlayingCardClassifier import PlayingCards_Dataset, PlayingCard_Classifier, transform
from scipy.signal import find_peaks
# import VideoStream

RANK = ['Ace', 'King', 'Queen', 'Jack', '10', '9','8','7','6','5','4','3','2']
SUIT = ['Hearts', 'Spades', 'Clubs', 'Diamonds']        

Card = namedtuple('Card', ['rank', 'suit']) # Class defined with namedtuple 

def makeDeck(deck, exclude_cards=None):
    ''' Returns all cards in the game expect given cards'''
    returnDeck = [] 
    if exclude_cards is None:
        exclude_cards = []
    else:
        x = 1 
        for card in deck:                
            if card not in exclude_cards:
                returnDeck.append(card)
                # print(f"Card #{x}: {card.rank} of {card.suit}"); x += 1
    return returnDeck

def deal_hand(deck, n):
        ''' Returns a random hand of "n" cards'''
        return random.sample(deck,n)
    
def get_rank_value(rank):
    ''' Returns a numerical value for the input rank. The heirarachy of the rank is defined in "rank".'''
    key_list = len(RANK) - RANK.index(rank)
    return key_list   

def evaluate_hand(hole_hand, openCards):
    ''' Evaluates the hand and other open cards (i.e. River + two flops). '''
        
    all_cards = hole_hand + openCards

    if len(all_cards) != 7:
        print('Error in "evaluate_hand": Total cards evaluated must be 7!')
        exit()
    elif len(hole_hand) != 2:
        print('Error in "evaluate_hand": Number of Hole Hands must be 2!')
        exit()
    
    suits_all_cards = []; ranks_all_cards = []; x = 1
    for card in all_cards:
        suits_all_cards.append(card.suit)
        ranks_all_cards.append(card.rank)

    rank_list = [] # Will be filled with the rank list 

    # initialization 
    pair = []; pair_multiplier = 0
    three_kind = []; four_kind = []; 
    strght = []; rank_list = []

    # Flush
    flush = []; flush_cards = []
    for suit in SUIT:
        flush.append(suits_all_cards.count(suit) >= 5)        
        if suits_all_cards.count(suit) >= 5:
            for card in all_cards:
                if card.suit == suit:
                    flush_cards.append(card)
                    rank_list.append(get_rank_value(card.rank))
    has_flush = any(flush)
    
    # check if Royal Flush or Straight Flush 
    has_strFlush = []; has_rylFlush = []; 
    if has_flush:        
        rank_list = sorted(rank_list,reverse=True); rank_list = rank_list[:5]
        if all(np.diff(rank_list)==np.diff(rank_list)[0]):
            if max(rank_list) == 13:
                has_rylFlush = True
            else:
                has_strFlush = True

    else:

        # Straight   
        for rank in ranks_all_cards:
            rank_list.append(get_rank_value(rank))    
        rank_list = np.array(sorted(rank_list, reverse=True)); rank_list = rank_list[:5]        
        if all(np.diff(rank_list)==np.diff(rank_list)[0]):
            strght = True

        # Check for pairs 
        else:            
            for rank in set(RANK):
                pair.append(ranks_all_cards.count(rank) == 2)
                if sum(pair) > 2: pair_multiplier = 2
                three_kind.append(ranks_all_cards.count(rank) == 3)
                four_kind.append(ranks_all_cards.count(rank) == 4)    

    # Scoring 
    score = 0 
    if has_rylFlush:
        score += 250
    elif has_strFlush:
        score += 200 
    elif any(four_kind):
        score += 175 
    elif any(three_kind) and any(pair):
        score += 150
    elif has_flush: 
        score += 100
    elif strght:
        score += 80
    elif any(three_kind):
        score += 40
    elif any(pair): 
        score += pair_multiplier*15

    score += sum(rank_list)

    return score
            
class calculate_PreFlop_Odds:
    ''' ============ claculate_PreFlop_Odds =================
    The objective of this class is to calculate the odds of a given hand using Monte-Carlo simulations. Broadly, the objective of this function is to: (1) Deal a set of random hands, (2) Evaluate the strength of each hand and (3) Calculate the winning probabilities
    '''

    def __init__(self):
        self.ranks = RANK
        self.suits = SUIT
        self.deck = [] 
        for suit in self.suits:
            for rank in self.ranks:
                self.deck.append(Card(rank, suit))    
   

    def simulate_MC_for_odds(self, hero_hand, numOpp, num_sims=10000):
        ''' Returns the results from the MC simulation'''
        wins = 0; splits = 0; c = 0 
        while (c <= num_sims):            
            
            # Creat a deck without the drawn cards 
            simDeck = makeDeck(self.deck, hero_hand)
            
            # Draw two more cards to emulate opponents hand
            simDeck1 = []
            oppHand_all = deal_hand(simDeck, numOpp*2)
            for card in simDeck:
                if card not in oppHand_all:
                    simDeck1.append(card)

            # Deal River + Flop
            openCards = deal_hand(simDeck1,5)
            # print('Hero Hand:',end=''); print(hero_hand)
            
            # Evaluate Score 
            # print('Preflop (hero hand + open cards)!')
            opp_score = []            
            hand_score = evaluate_hand(hero_hand, openCards); 
            
            nOpTot = numOpp*2

            for nO in range(0, nOpTot, 2):  
                # print(f'Preflop (opponent hand + open cards) #{nO}!')                
                opp_score.append(evaluate_hand(oppHand_all[nO:nO+2], openCards));          
            
            # Determine Winners 
            allScores = [hand_score] + opp_score; maxScore = max(allScores)
            numWinners = 0; 
            if hand_score == maxScore:
                for score in allScores:
                    if score == maxScore:                
                        numWinners += 1
                if numWinners == 1:
                    wins += 1                          
                else:
                    splits += 1/numWinners # Splits equally among winners 

            # Print Statements for Debugging 
            # print(f'#{c}: Hand = [{hero_hand[0].suit} {hero_hand[0].rank}, {hero_hand[1].suit} {hero_hand[1].rank}], River = [', end='')
            # for card in openCards:
            #     print(f'{card.suit} {card.rank}, ', end='')
            # print(']')
            


            c += 1

        # print(f'Total Wins: {wins}, Total Splits: {splits}')
        win_rate = wins/num_sims
        split_rate = splits/num_sims

        

        return  win_rate, split_rate

class calculate_PostFlop_Odds:
    ''' ============ claculate_PostFlop_Odds =================
    The objective of this class is to calculate the odds of a known hand after the flop (i.e. revealing of a river) using Monte-Carlo simulations. 
    '''
    def __init__(self):
        self.ranks = RANK
        self.suits = SUIT
        self.deck = [] 
        for suit in self.suits:
            for rank in self.ranks:
                self.deck.append(Card(rank, suit))

    def simulate_MC_for_odds(self, hero_hand, commHand, numOpp, num_sims=10000):
        ''' Returns the results from the MC simulation'''
        wins = 0; splits = 0; c = 0 
        while (c <= num_sims):            
            
            # print('Post Flop:\n')

            heroComm_hand = hero_hand + commHand

            # Creat a deck without the drawn cards 
            simDeck = makeDeck(self.deck, heroComm_hand)
            
            # Draw two more cards to emulate opponents hand
            simDeck1 = []
            oppHand_all = deal_hand(simDeck,numOpp*2)
            
            for card in simDeck:
                if card not in oppHand_all:
                    simDeck1.append(card)
            
            # Deal Flop
            openCards = deal_hand(simDeck1,2)
            # print('Open Cards: ', end=''); print(commHand+openCards); print('\n')
            
            # Evaluate Score 
            opp_score = []            
            hand_score = evaluate_hand(hero_hand, openCards+commHand)   
            # print('Hand Cards: ', end=''); print(hero_hand)
            # print(f'\t Score: {hand_score}\n')
            
            nOpTot = numOpp*2

            for nO in range(0, nOpTot, 2):                
                score = evaluate_hand(oppHand_all[nO:nO+2], openCards+commHand)
                opp_score.append(score); 
            #     print(f'Opp. #{nO/2} Cards: ', end=''); print(oppHand_all[nO:nO+2], end=' ')
            #     print(f'----- Score: {score}')
            # print('\n')
                
            # Determine Winners 
            allScores = [hand_score] + opp_score; maxScore = max(allScores)
            numWinners = 0; 
            if hand_score == maxScore:
                for score in allScores:
                    if score == maxScore:                
                        numWinners += 1
                if numWinners == 1:
                    wins += 1                          
                else:
                    splits += 1/numWinners # Splits equally among winners 

            ## Print statements for debugging   
            # if numWinners == 0:
            #     print(f'#{c}: Hand = [{hero_hand[0].suit} {hero_hand[0].rank} ({get_rank_value(hero_hand[0].rank)}), {hero_hand[1].suit} {hero_hand[1].rank} ({get_rank_value(hero_hand[1].rank)})], River = [', end='')
            #     for card in openCards+commHand:
            #         print(f'{card.suit} {card.rank} ({get_rank_value(card.rank)}), ', end='')
            #     print(']')              
            #     for nO in range(0, nOpTot, 2):  
            #         print(f'Opp. #{int(nO/2)}: [{oppHand_all[nO].suit} {oppHand_all[nO].rank} ({get_rank_value(oppHand_all[nO].rank)}), {oppHand_all[nO+1].suit} {oppHand_all[nO+1].rank} ({get_rank_value(oppHand_all[nO+1].rank)})] ---- score: {opp_score[int(nO/2)]}')
            #     print(f'Scores: Hand = {hand_score}, Opp = {opp_score}, Max = {maxScore}\n')

                     
            # print(f'#{c}: (Win,numWins) = ({wins}, {numWinners})')
            
            c += 1

        # print(f'Total Wins: {wins}, Total Splits: {splits}')
        win_rate = wins/num_sims
        split_rate = splits/num_sims

        return  win_rate, split_rate

def find_cards(frame, camDim): 
    ''' ================= OpenCamera ================
    The objective of this class is to process the recorded frame of card classification.   '''   

    # Image transforms 
    frameBlur = cv2.GaussianBlur(frame, (3,3), 5,5)         # Apply Blur 
    frameHSV = cv2.cvtColor(frameBlur, cv2.COLOR_RGB2HSV)   # to HSV (detects whites better) 
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)     # to gray for classifier 
    frameDim = frame.shape                                  # For checking card areas 

    # Bounds for white detection 
    sensitivity = 85                              # Adjust sensitivity based on current lighting
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    # Find Card Contours 
    mask = cv2.inRange(frameHSV, lower_white, upper_white) # get mask 
    cnts,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours 
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True) # Arrange contours in descending order of area 
    
    
    # Pick top 7
    vertLoc = int(camDim[0]*0.5) # checking the lower 50% of the screen for hand cards 
    cnts_sort = []; cardAreas = []; handCards_idx = []; cnts_final = []; vertF = []

    if len(index_sort) >= 7: tIter = 7
    else: tIter = len(index_sort)
    
    if not cnts: 
        print('No contours found!')
    else:
        for idx in range(tIter): #range(len(index_sort)):                  # Loop through contours 
            cnts_sort.append(cnts[index_sort[idx]])         # Sorted contour list 
            verts = cv2.boundingRect(cnts_sort[idx])        # Get verticies of contour 
            x,y,w,h = verts[0],verts[1],verts[2],verts[3]; vertF.append(verts)
            contArea = cv2.contourArea(cnts_sort[idx])
            
            if contArea/(frameDim[0]*frameDim[1]) > 0.5e-2:
                cardAreas.append(frameGray[y:y+h, x:x+w])    # Images within final contours
                cnts_final.append(cnts_sort[idx])           # Track final contours
                
                # Find the number of 
                ct = 0; arr = range(y, y+h)
                for val in arr:
                    if val > vertLoc: ct += 1               # zero is upper left corner of the screen
                if ct > len(arr)/2: handCards_idx.append(True)
                else: handCards_idx.append(False)
                # if max(set(checkArr), key=checkArr.count):
            # if len(cardAreas) == 7:
            #     break

    # Draw Contours  
    frame_wConts = cv2.drawContours(frame, cnts_final, -1, (255,0,0), 2)
    frame_wConts = cv2.line(frame_wConts, (0,vertLoc), (camDim[1],vertLoc), color=(0,0,0), thickness=2)   

    # return a frame that can be concatinated to the original frame that is being plotted to the figure. 

    return frame_wConts, cardAreas, handCards_idx, vertF        

# Function for on camera buttons

def button_callback(event, x, y, flags, param):
    global computeButton, resetSim, isDisplayed, findCard
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within button boundaries        
        if buttonPos[0] <= x <= buttonPos[2] and buttonPos[1] <= y <= buttonPos[3]:        
            if not isDisplayed:
                computeButton = True; findCard = False
            else: 
                computeButton = False; resetSim = True 

def button_action(frame, bn):
    ''' Displays the button and associated changes '''
    global resetSim
    # Draw button (rectangle)
    xTemp = []
    if bn == 0:
        xTemp = 0
        if computeButton:
            button_color = (169, 169, 169) 
            buttonText = 'Computing...'; xTex = buttonPos[3]+20
        else:
            button_color = (0, 0, 0)
            buttonText = 'Compute Odds?'; xTex = buttonPos[3]+10           

    elif bn == 1:
        xTemp = 30
        button_color = (0, 0, 0) 
        buttonText = 'Compute New Odds?'; xTex = buttonPos[3]-22
        if computeButton:
            resetSim = True

    edgeThick = 2
    cv2.rectangle(frame, (buttonPos[0]-edgeThick-xTemp, buttonPos[1]-edgeThick), (buttonPos[2]+edgeThick, buttonPos[3]+edgeThick), (255,255,255), -1)    
    cv2.rectangle(frame, (buttonPos[0]-xTemp, buttonPos[1]), (buttonPos[2], buttonPos[3]), button_color, -1)
    
    # Add text to button
    cv2.putText(frame, buttonText, (xTex, buttonPos[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def displayTable(frame):
    ''' Print the table to display resutls
    
     Arguments:
        frame    - current frame to place the table on        
    '''
    # Printing table
    tabelPos = [50, 100, 600, 380]
    
    tableFrame = frame.copy()
    cv2.rectangle(tableFrame, (tabelPos[0], tabelPos[1]), (tabelPos[2], tabelPos[3]), (0,0,0), -1)   
    alpha = 0.5; frame = cv2.addWeighted(tableFrame, alpha, frame, 1 - alpha, 0); 
    cv2.rectangle(frame, (tabelPos[0], tabelPos[1]), (tabelPos[2], tabelPos[3]), (255,255,255), 2)  
    xLine = 160 + tabelPos[0]; cv2.line(frame, (xLine,tabelPos[1]), (xLine,tabelPos[3]), (255,255,255), 2)
    yLine = 65 + tabelPos[1]; cv2.line(frame, (tabelPos[0],yLine), (tabelPos[2],yLine), (255,255,255), 2)

    # First Block 
    txtArray = ['Number', 'of Players', 'Win', 'Rate (%)', 'Split', 'Rate (%)', 'Total', 'Equity (%)']
    spc = [20, 45, 20, 45, 20, 45, 20, 45]; xs = 0; yTex, xTex = tabelPos[1] + 30, tabelPos[0] + 10

    for txt in txtArray: 
        cv2.putText(frame, txt, (xTex, yTex), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); 
        if xs >= 2:
            if xs % 2: cv2.putText(frame, 'Post', (xTex+105, yTex), cv2.FONT_HERSHEY_PLAIN, 1, (118, 244, 249), 2); 
            else: cv2.putText(frame, 'Pre', (xTex+105, yTex), cv2.FONT_HERSHEY_PLAIN, 1, (118, 244, 249), 2); 
        yTex += spc[xs]; xs += 1

    return frame, tabelPos

def displayResults(frame, preFlop, postFlop, maxPlyrs, tablePos):
    ''' Prints the table with results. 
    Asumes that a table has been displayed (with display Table)
    
    Arguments:
        frame    - current frame to place the table on 
        preFlop  - array of Preflop win and split rate 
        postFlop - array of Postflop win and split rate 
        maxPlyrs - maximum number of players to be displayed
    '''    
    if len(preFlop) != len(postFlop):
        print('Number of PreFlop and PostFlop data must match the number of values in nPlyrs!'); exit()

    nPlyrs = range(3,maxPlyrs+1) # This function is optimized for printing 

    # Other Blks
    blk0 = 180; blkSize = int((tablePos[2]-tablePos[0]-blk0)/len(nPlyrs)); lineSpace = 65

    for blk in range(len(nPlyrs)):        
        xTex, yTex = tablePos[0] + blk0 + blkSize*(blk), tablePos[1] + 40
        
        spc = 12
        cv2.putText(frame, f" {nPlyrs[blk]}", (xTex, yTex), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); yTex += lineSpace;         
                
        if blk < len(preFlop):

            cv2.putText(frame, f"{preFlop[blk][0]*100:.1f}", (xTex, yTex-spc), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); 
            cv2.putText(frame, f"{postFlop[blk][0]*100:.1f}", (xTex, yTex+spc), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); yTex += lineSpace

            cv2.putText(frame, f"{preFlop[blk][1]*100:.1f}", (xTex, yTex-spc), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); 
            cv2.putText(frame, f"{postFlop[blk][1]*100:.1f}", (xTex, yTex+spc), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); yTex += lineSpace

            cv2.putText(frame, f"{(preFlop[blk][0]+preFlop[blk][1])*100:.1f}", (xTex, yTex-spc), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); 
            cv2.putText(frame, f"{(postFlop[blk][0]+postFlop[blk][1])*100:.1f}", (xTex, yTex+spc), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); yTex += lineSpace

        # elif nPlyrs[blk] < cPlyr: continue
        # else:
        #     cv2.putText(frame, f"???", (xTex, yTex), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2); 
    
        # NOTE: Iteration through the loops is incorrect! Start debugging with this!            

    
    return frame

# input my cards
# Use of manual input of cards only         
def cardID_to_ClassText(text):
    ''' Translated Card ID to Text and Rank'''
    
    if len(text) == 3: rank, suit = text[0],text[2]
    else: rank, suit = text[0],text[1]
    
    if suit.casefold() == 'h':
        suit = 'Hearts'
    elif suit.casefold() == 's':
        suit = 'Spades'
    elif suit.casefold() == 'c':
        suit = 'Clubs'
    elif suit.casefold() == 'd':
        suit = 'Diamonds' 

    if rank.casefold() == 'k':
        rank = 'King'
    elif rank.casefold() == 'q':
        rank = 'Queen'
    elif rank.casefold() == 'j':
        rank = 'Jack'
    elif rank.casefold() == 'a':
        rank = 'Ace' 
    elif rank.casefold() == '1':
        rank = '10'
    else:
        rank
    return rank, suit


    
# =================== Main Function =====================
def main():
    preFlop = calculate_PreFlop_Odds()
    postFlop = calculate_PostFlop_Odds()
    
    model = PlayingCard_Classifier()
    model.load_state_dict(torch.load('Cards_Dataset2/CardClassifier.pth'))
    model.eval()
    classLabels = cardNN.get_classLabels('Cards_Dataset2/')
    

    # Initialize Camera 
    cap = cv2.VideoCapture(1) # using an external camera
    if not cap.isOpened():
        print("Error: Could not open Camera.")
        return            
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'Frame rate of accessed camera = {fps}')
    if fps == 0:
        fps = 30  # fallback to 30 fps if camera doesn't report fps
    
    # Button info 
    cv2.namedWindow('Looking for cards...')
    cv2.setMouseCallback('Looking for cards...', button_callback)
    global buttonPos, computeButton, resetSim, findCard, isDisplayed
    buttonPos = [frame_width-190, frame_height-80, frame_width-40, frame_height-30]

    # Open Camera 
    runCam = True; isDisplayed = False; findCard = True
    computeButton = False; resetSim = False; bn = 0
    handCard = []; riverCard = []; nPl = 0; frameLst = []
    preFlopFinal = []; postFlopFinal = []; 
    try:
        while runCam:
            
            # Read Camera data 
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera!"); break        

            # Temporary containers that hold cards 
            handCardTemp = []
            riverCardTemp = []               
            
            if findCard:
                # ================ Idenifying cards ======================
                frame, cardAreas, handCard_idx, verts = find_cards(frame, (frame_height, frame_width))  
                # print(f'Total Cards detected: {len(cardAreas)}')      

                if len(cardAreas) > 0:
                    # Transform cards for model
                    data = torch.zeros((32,1,128,128))
                    for idx in range(len(cardAreas)):                        

                        frameTensor = Image.fromarray(cardAreas[idx]) # Numpy to PIL Image
                        data[idx,0,:,:] = transform(frameTensor)


                    score = model(data)             # Model predicition 
                    _, prediction = score.max(1)                
                    
                    for idx in range(len(handCard_idx)): 
                        classLabel = str(classLabels[prediction[idx].item()])                    
                        # print(f'Card #{idx}: {classLabel}') 

                        rank, suit = cardID_to_ClassText(classLabel)

                        # Place text identifying card on the card area
                        card_info = rank + ' ' + suit
                        frame = cv2.putText(frame, card_info, (verts[idx][0],verts[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (238,238,155), 2, cv2.LINE_AA)

                        if handCard_idx[idx]:                    
                            handCardTemp.append(Card(rank,suit))
                            
                        else: 
                            riverCardTemp.append(Card(rank,suit)) 
                    
                    handCard = handCardTemp; riverCard = riverCardTemp
                        
            else: 

                if not handCard or not riverCard:
                    print('Hand Cards: ', end=''); print(handCard)
                    print('River Cards: ', end=''); print(riverCard)
                    frame = cv2.putText(frame, 'NO CARDS FOUND!', (verts[idx][0],verts[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (37,150,190), 2, cv2.LINE_AA)
                    print('No cards found! Returning to camera feed...')
                    findCard = True

                else:                     
                    # ============= Computing probabilities ============================                    
                    if computeButton:
                                                
                        nPlyrs = range(3,8)

                        print(f'\nComputing porbabilities for {nPlyrs[nPl]} opponents...\nHand Cards: {handCard[0].rank} of {handCard[0].suit}, {handCard[1].rank} of {handCard[1].suit}')
                        print(f'River Cards: {riverCard[0].rank} of {riverCard[0].suit}, {riverCard[1].rank} of {riverCard[1].suit}, {riverCard[2].rank} of {riverCard[2].suit}')

                        # Drawing Table to display results 
                        frame, tablePos = displayTable(frame);                         

                        preFlopRes = preFlop.simulate_MC_for_odds(handCard, numOpp=nPlyrs[nPl], num_sims=1e4)                                 # [Win Rate, Split Rate]
                        pstFlopRes = postFlop.simulate_MC_for_odds(handCard, riverCard, numOpp=nPlyrs[nPl], num_sims=1e4)      # number of sims - 1e4 

                        preFlopFinal.append([preFlopRes[0], preFlopRes[1]]); 
                        postFlopFinal.append([pstFlopRes[0], pstFlopRes[1]])

                        frame = displayResults(frame, preFlopFinal, postFlopFinal, 7, tablePos)
                        
                        output = [['Win Rate', preFlopRes[0]*100, pstFlopRes[0]*100], ['Split Rate', preFlopRes[1]*100, pstFlopRes[1]*100], ['Total Equity', (preFlopRes[0]+preFlopRes[1])*100, (pstFlopRes[0]+pstFlopRes[1])*100]]
                        print(tabulate(output, headers=('','Pre-Flop','Post-Flop'), floatfmt=".1f"))                                              
                        
                        nPl += 1
                        if nPl == len(nPlyrs):
                            frameLst = frame
                            print('\n***** Completed all runs!'); isDisplayed = True;  computeButton = False
                            
                    else:
                        frame = frameLst; bn = 1
                        cv2.putText(frame, 'Press "q" to quit!"', (15, frame_height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        if resetSim:
                                print('**** Simulation has been reset! ****')
                                isDisplayed = False; computeButton = False; bn = 0
                                resetSim = False; findCard = True; 
                                handCard = []; riverCard = []; nPl = 0; frameLst = []
                                preFlopFinal = []; postFlopFinal = []; 

            # Display button 
            frame = button_action(frame, bn)

            # frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)


            cv2.imshow('Looking for cards...', frame)
    

            # break if 'q' is pressed 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                runCam = False       

    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()