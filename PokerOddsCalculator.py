# ========== Poker Hands ==========
# The objective of this program is to calculate the odds of a given poker hand. We will access the device's camera, identify the cards in play and then proceed to print out the odds of winning. 

from collections import namedtuple
from PIL import Image 
import random, torch
import matplotlib.pyplot as plt 
import re, cv2, numpy as np
from tabulate import tabulate
import trainNN_cardClassifier_v1 as cardNN
from trainNN_cardClassifier_v1 import PlayingCards_Dataset, PlayingCard_Classifier, transform
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
            
            # Evaluate Score 
            opp_score = []            
            hand_score = evaluate_hand(hero_hand, openCards)
            
            nOpTot = numOpp*2

            for nO in range(0, nOpTot, 2):                
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
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # to RGB 
    frameBlur = cv2.GaussianBlur(frameGray, (15,15), 30,30)

    # Find Card Contours 
    mask = cv2.inRange(frameBlur,180,255) # get mask 
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Find contours 
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True) # Arrange contours in descending order of area 
    
    # Pick top 7
    vertLoc = int(camDim[0]*0.8) # checking the lower 20% of the screen for hand cards 
    cnts_sort = []; cardAreas = []; handCards_idx = []; searchCards = True
    
    if not cnts: 
        print('No contours found!')
    else:
        for idx in range(len(cnts)):                    # Loop through contours 
            cnts_sort.append(cnts[index_sort[idx]])     # Sorted contour list 
            x,y,h,w = cv2.boundingRect(cnts_sort[idx])  # Get verticies of contour 
            contArea = frameGray[y:y+h, x:x+w]

            hist = cv2.calcHist(contArea, [0], None, [256], [0, 256])    
            peaks, _ = find_peaks(np.array(hist).flatten(), height=(5000,75000))
            
            if len(peaks[:5]) >= 2:
                if y > vertLoc: handCards_idx.append(True)
                else: handCards_idx.append(False)

                cardAreas.append(contArea) # Card areas                    

    # Draw Contours  
    frame_wConts = cv2.drawContours(frame, cnts_sort, -1, (255,0,0), 5)
    frame_wConts = cv2.line(frame_wConts, (0,vertLoc), (camDim[1],vertLoc), color=(255,0,0), thickness=2)   

    # return a frame that can be concatinated to the original frame that is being plotted to the figure. 

    return frame_wConts, cardAreas, handCards_idx        



# input my cards
# Use of manual input of cards only         
def cardID_to_ClassText(text):
    ''' Translated Card ID to Text and Rank'''
    rank, suit = text[0],text[1]
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
    elif rank.casefold() == 'a' or rank == '1':
        rank = 'Ace' 
    elif rank.casefold() == 't':
        rank = '10'
    else:
        rank
    return rank, suit


# # Define transforms
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize to standard size
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                     std=[0.229, 0.224, 0.225])  # ImageNet normalization
# ])

    
# =================== Main Function =====================
def main():
    preFlop = calculate_PreFlop_Odds()
    postFlop = calculate_PostFlop_Odds()
    
    model = torch.load('Cards_Dataset/CardClassifier.pth')
    classLabels = cardNN.get_classLabels('Cards_Dataset/')
    
    # Card initialization 
    handCard = []
    riverCard = []

    # Initialize Camera 
    cap = cv2.VideoCapture(0)
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

    # Open Camera 
    runCam = True; 
    try:
        while runCam:
            
            # Read Camera data 
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera!"); break
            
            # ================ Idenifying cards ======================
            frame, cardAreas, handCard_idx = find_cards(frame, (frame_height, frame_width))        

            if any(cardAreas):
                # Transform cards for model
                data = torch.zeros((32,1,128,128))
                for idx in range(len(cardAreas)):
                                
                    frameTensor = Image.fromarray(cardAreas[idx]) # Numpy to PIL Image
                    data[idx,0,:,:] = transform(frameTensor)

                score = model(data)             # Model predicition 
                _, prediction = score.max(1)
                
                for idx in range(len(handCard_idx)): 
                    classLabel = str(classLabels[prediction[idx].item()]); 
                    rank, suit = cardID_to_ClassText(classLabel)
                    if handCard_idx[idx]:                    
                        handCard.append(Card(rank,suit))
                    else: 
                        riverCard.append(Card(rank,suit)) 
            else:                           
                classLabel = 'No Card Found!'                

            frame = cv2.putText(frame, str(classLabel), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Looking for cards...', frame)


            x = 0
            # for card_idx in handCard_idx:
            #     card = cardAreas[x]; x += 1
            #     if card_idx:
                    
            # ========================================================        

            # break if 'q' is pressed 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                runCam = False       

    finally:
        cap.release()
        cv2.destroyAllWindows()

    






    # while run:
        
    #     # # my hand     
    #     # print('Enter Hand Cards:')
    #     # rank1, suit1 = input_myCards(1)
    #     # rank2, suit2 = input_myCards(2)
    #     # hand = [Card(rank1,suit1), Card(rank2,suit2)]; 

    #     # print('Enter River?', end=' '); text = input()
    #     # if text == 'y':
    #     #     # # River Cards         
    #     #     print('Enter River Cards:')
    #     #     rank1, suit1 = input_myCards(1)
    #     #     rank2, suit2 = input_myCards(2)
    #     #     rank3, suit3 = input_myCards(3)
    #     #     river = [Card(rank1,suit1), Card(rank2,suit2), Card(rank3,suit3)]; 
        
    #     # hand = [Card('6','Spades'), Card('2','Diamonds')]
    #     # river = [Card('3','Hearts'), Card('4','Clubs'), Card('5','Clubs')]

    #     hand = [Card('2','Spades'), Card('5','Spades')]
    #     river = [Card('Ace','Hearts'), Card('10','Spades'), Card('Jack','Spades')]

        
    #     # score = evaluate_hand(hand, river+[Card('9','Hearts'), Card('4','Spades')])
    #     # print(score)
    #     # run = False 
        

    #     # print(hand)
    #     # Win probability
        
    #     print(f'\nComputing porbabilities...\nHand Cards: {hand[0].rank} of {hand[0].suit}, {hand[1].rank} of {hand[1].suit}')
    #     print(f'River Cards: {river[0].rank} of {river[0].suit}, {river[1].rank} of {river[1].suit}, {river[2].rank} of {river[2].suit}')
    #     for nOps in [6]:#range(1,2):
    #         wR_preFlop, sR_preFlop = preFlop.simulate_MC_for_odds(hand, numOpp=nOps, num_sims=1e4)
    #         wR_pstFlop, sR_pstFlop = postFlop.simulate_MC_for_odds(hand, river, numOpp=nOps, num_sims=1e4) #1e4

            
    #         print(f"\nNumber of opponents: {nOps}\n")
    #         # print(f"Win Rate: {wR_preFlop:.1%}")
    #         # print(f"Split Rate: {sR_preFlop:.1%}")
    #         # print(f"Total equity: {(wR_preFlop + sR_preFlop):.1%}")
    #         output = [['Win Rate', wR_preFlop*100, wR_pstFlop*100], ['Split Rate', sR_preFlop*100, sR_pstFlop*100], ['Total Equity', (wR_preFlop+sR_preFlop)*100, (wR_pstFlop+sR_pstFlop)*100]]
    #         print(tabulate(output, headers=('','Pre-Flop','Post-Flop'), floatfmt=".1f"))
    
    #     print('Continue Calculating? (y/n): ', end=''); text = input()
    #     if text == 'n':
    #         run = False            




if __name__ == '__main__':
    main()