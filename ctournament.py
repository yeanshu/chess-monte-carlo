import random
import csv
import copy
import os
from collections import Counter
import itertools

def add_to_dict(dict, val, add):
    if val in dict:
        dict[val] += add
    else:
        dict.setdefault(val, add)

def clamp(val):
    return max(min(val,1),0)

def elo_util(white,black, k=24):
    dif = white-black
    expected = clamp(0.54313+0.0011064*dif)
    win = k*(1-expected)
    draw = k*(0.5-expected)
    loss = k*(-expected)
    return expected

def update_elo(rates):
    players = {}
    with open(rates) as f:
        reader = csv.reader(f)
        for row in reader:
            rating = float(row[1])
            score = float(row[2])
            games = [c for c in row[3:] if c !='']
            players[row[0]] = [rating,score,games]

    for player, games in players.items():
        ratinga = games[0]
        score = games[1]
        exp_score = 0
        for game in games[2]:
            colour = game[0]
            ratingb = int(game[1:])
            expect = 0
            if colour == 'w':
                expect = elo_util(ratinga,ratingb)
            else:
                expect = 1-elo_util(ratingb,ratinga)
            exp_score += expect
        change = 24*(score-exp_score)
        #games[0] = round(ratinga+change)
        #players[player] = round(change,1)
        players[player] = round(ratinga+change)
        #games.pop()
        #games[1] = round(change,1)

    players = dict(sorted(players.items(), key=lambda item: -item[1]))
    print(players)
    return

def getodds(white, black, tformat=0):
    #Draw Rates
    time_adj = [0.68,0.52,0.44]
    dif = white-black
    #score = math.erfc(dif / ((2000/7) * math.sqrt(2))) / 2 NORMAL
    #score = 1/(1+10**(dif/400)) #LOGISTIC
    score = clamp(0.54313+0.0011064*dif) #LINEAR
    dr = abs(dif+34)
    dr = -3.2845E-06*(dr**2) - 0.0012031*dr + 1
    time_adj = [0.68,0.52,0.44]
    drawrate= dr*time_adj[tformat]
    winrate = clamp(score - (drawrate/2))
    loserate = clamp(1-winrate-drawrate)
    return [winrate,drawrate,loserate]

def simulate_player_game(p1, p2, up=True, reps=1,tformat=0):
    target = getodds(p1[0], p2[0],tformat=tformat)
    luck = random.random()
    score = 0
    if luck > 1-target[2]:
            #Black wins
            if up:
                p2[1] += 1
            return 1

    elif luck < target[0]:
            #White wins
            if up:
                p1[1] += 1
            return 0

    else:
            #Draw
            if up:
                p1[1] += 0.5
                p2[1] += 0.5
            return 0.5
    

def simple_simulate(p1, p2, rep=1):
    for i in range(rep):    
        if i%2 == 0:
            target = getodds(p1, p2)
        else:
            target = getodds(p2, p1)
            target.reverse()

        luck = random.random()
        if luck < target[0]:
            print("P1 Wins")

        elif luck < target[0]+target[1]:
            print("Draw")
        else:
            print("P2 Wins")

def simulate_matches(data, matches):
    new_data = data
    for n in range(0,len(matches)):
        p1 = new_data[matches[n][0]]
        p2 = new_data[matches[n][1]]
        simulate_player_game(p1, p2)
    return new_data

def simulate_swiss(data, rounds=11, tformat=0):
    new_data = data
    pairs = generate_pairs(new_data)
    for _ in range(rounds):
        for pair in pairs:
            i = round(random.random())
            simulate_player_game(new_data[pair[i]],new_data[pair[1-i]],tformat=tformat)
        pairs = generate_pairs(new_data)
    return new_data

def generate_pairs(data):

    pairs = []
    scores = [data[d][1] for d in data]
    scores = Counter(scores)
    scores = dict(sorted(scores.items(), key=lambda item: -item[0]))
    backup = None
    for score in scores:
        new = []
        pos = [d for d in data if data[d][1] == score]
        if backup:
                pos.insert(0,backup)
        size = len(pos)
        if size % 2 == 0:
            l1 = pos[:size//2]
            l2 = pos[size//2:]
            new = list(zip(l1,l2))
            pairs.extend(new)
            backup = None
        else:
            l1 = pos[:size//2]
            l2 = pos[size//2:-1]
            new = list(zip(l1,l2))
            pairs.extend(new)
            backup = pos[-1]
    return pairs

def find_winner(data, tiebreak=True, top=1,verbose=False):
    score = [(k, v[1]) for k, v in data.items()]
    score.sort(key=lambda tup: -tup[1])
    win_score = score[0][1]
    if top != 1:
        top_score = score[top-1][1]
        good = [p for p in score if p[1] > top_score]
        pot = [p for p in score if p[1] == top_score]
        remain = top-len(good)
        pot = random.sample(pot,remain)
        good.extend(pot)
        return good
    if score[0][1] == score[1][1]:
        #Tiebreaks
        args = [p for p in score if p[1] == win_score]
        args = random.sample(args,2)
        target = getodds(data[args[0][0]][0], data[args[1][0]][0])
        while tiebreak:
            luck = random.random()
            if luck < target[0]:
                return [args[0]]
            elif luck > target[0]+target[1]:
                return [args[1]]

        names = [p[0] for p in score if p[1] == win_score]
        ties = len(names)
        names = '-'.join(names)

        if verbose:
            return [(names,score[-1][1])]
        return [(str(ties) + '-way Draw', score[-1][1])]
    return [score[0]]

def monte_carlo(n, data, matches, tiebreak=True, top=1, verbose=False, prob=True, format='',score_dist=None):
    winners = dict.fromkeys(list(data.keys()),0)
    scores = dict.fromkeys(list(data.keys()),0)
    scores_dist = {}
    win_score_dist = {}
    original_data = copy.deepcopy(data)
    for _ in range(n):
        cur_data = {}
        match format:
            case 'swiss':
                cur_data = simulate_swiss(data)
            case 'knockout':
                cur_data = generate_knockout(data)
            case 'custom':
                cur_data = simulate_matches(data, matches)
            case 'extra':
                cur_data = simulate_extra(data)
            case 'grandprix':
                cur_data = simulate_prix(data,matches)
            case _:
                return
        tops = find_winner(cur_data,tiebreak,top,verbose)
        for t in tops:
            add_to_dict(winners,t[0], 1) 
        add_to_dict(win_score_dist,tops[-1][1],1)               
        for k, _ in cur_data.items():
            add_to_dict(scores,k,cur_data[k][1])
        if score_dist:
            add_to_dict(scores_dist, cur_data[score_dist][1], 1)
        data = copy.deepcopy(original_data)
    winners = dict(sorted(winners.items(), key=lambda item: -item[1]))
    if prob:
        winners.update({k: v/n*100 for k, v in winners.items()})
    scores = dict(sorted(scores.items(), key=lambda k: -winners[k[0]]))
    scores.update({k: round(v/n,2) for k, v in scores.items()})
    scores_dist = dict(sorted(scores_dist.items()))
    win_score_dist = dict(sorted(win_score_dist.items()))
    print(winners)
    # print(scores)
    # print(scores_dist)
    # print(win_score_dist, sum([k*v for k,v in win_score_dist.items()])/n)

def simulate_extra(data):
    new_data = data
    results = simulate_swiss(new_data)

    knockout = {}
    winners = find_winner(results,top=8)
    for w in winners:
        knockout[w[0]] = [new_data[w[0]][0], 0]

    return generate_knockout(knockout)

def simulate_prix(data, matches):
    prixgames = [prixa,prixc,prixd,prixb]

    new_data = data
    a = dict(itertools.islice(new_data.items(),4))
    b = dict(itertools.islice(new_data.items(),4,8))
    c = dict(itertools.islice(new_data.items(),8,12))
    d = dict(itertools.islice(new_data.items(),12,16))

    knockout = {}

    for n, group in enumerate([a,c,d,b]):
        matches = prixgames[n]
        results = simulate_matches(group,matches)
        winner = find_winner(results)[0]
        knockout[winner[0]] = [new_data[winner[0]][0],0]
        
    #return knockout
    return generate_knockout(knockout)

def generate_knockout(data, rounds=2,tformat=0):
    new_data = data
    remaining = list(data)
    while len(remaining) > 1:
        left = len(remaining)//2
        next = [] 
        for i,_ in enumerate(remaining[:left]):
            while True:
                res = simulate_player_game(data[remaining[i]], data[remaining[-1-i]], up=False,tformat=tformat)
                res += 1-simulate_player_game(data[remaining[-1-i]], data[remaining[i]], up=False,tformat=tformat)
                #print(remaining[i], remaining[-1-i], res)
                if res < 1:
                    next.append(remaining[i])
                    break
                if res > 1:
                    next.append(remaining[-1-i])
                    break
                res = 0
        remaining = next
    new_data[remaining[0]][1] += 1
    return new_data

data = {}
ori_data = {}
matches = []

def read_matches(fn):
    mchs = []
    with open(fn) as f:
        reader = csv.reader(f)
        mchs[:] = list(map(tuple, reader))
        mchs[:] = [(n[0], n[1]) for n in mchs if n[2] == '0']

    return mchs

def load_data(ratings, matchup):
    with open(ratings) as f:
        reader = csv.reader(f)
        for row in reader:
            row[1:] = map(float, row[1:])
            data[row[0]] = row[1:]
            ori_data[row[0]] = row[1:]

    with open(matchup) as f:
        reader = csv.reader(f)
        matches[:] = list(map(tuple, reader))
        matches[:] = [(n[0], n[1]) for n in matches if n[2] == '0']

prixa = read_matches('./grandprix/prixa.csv')
prixb = read_matches('./grandprix/prixb.csv')
prixc = read_matches('./grandprix/prixc.csv')
prixd = read_matches('./grandprix/prixd.csv')

os.chdir('C:/Users/Yean/Downloads/pem/Programs/Chess')
load_data('./grandprix/prixr.csv', './grandprix/prixa.csv')
monte_carlo(50000,data,matches, format='grandprix', tiebreak=True,verbose=True, top=1, prob=True, score_dist=None)
#update_elo('./ratings/2022Feb.csv')
#pgn_csv("./Test.pgn")

#simple_simulate(2760,2682, rep=2)
#print(getodds(data['Carlsen'][0], data['Praggnanandhaa'][0]))