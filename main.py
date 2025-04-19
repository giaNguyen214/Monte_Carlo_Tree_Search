import chess
import chess.pgn
import chess.svg
import chess.engine
import random
import time
from math import log,sqrt,e,inf
from io import BytesIO
import numpy as np
import requests
import pandas as pd
from PIL import Image
# from cairosvg import svg2png
import cv2
import streamlit as st

level = "easy"

def app():
    class node():
        def __init__(self):
            self.state = chess.Board()
            self.action = ''
            self.children = set()
            self.parent = None
            self.N = 0
            self.n = 0
            self.v = 0

    # def ucb1(curr_node):
    #     ans = curr_node.v+2*(sqrt(log(curr_node.N+e+(10**-6))/(curr_node.n+(10**-10))))
    #     return ans
    def ucb1(curr_node, level):
        if level == 'easy':
            c = 2.5  
        elif level == 'medium':
            c = 1.4
        else:
            c = 0.8  
        
        exploit = curr_node.v
        explore = c * (sqrt(log(curr_node.N+e+(10**-6))/(curr_node.n+(10**-10))))
        return exploit + explore


    def rollout(curr_node):
        #Checking whether the current position of the node is checkmate
        if(curr_node.state.is_game_over()):
            board = curr_node.state
            if(board.result()=='1-0'):
                #print("h1")
                # st.header("Checkmate! White wins")
                return (1,curr_node)
            elif(board.result()=='0-1'):
                # st.header("Checkmate! Black wins")
                #print("h2")
                return (-1,curr_node)
            else:
                return (0.5,curr_node)
        
        all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
        
        for i in all_moves:
            tmp_state = chess.Board(curr_node.state.fen())
            tmp_state.push_san(i)
            child = node()
            child.state = tmp_state
            child.parent = curr_node
            curr_node.children.add(child)
        rnd_state = random.choice(list(curr_node.children))

        return rollout(rnd_state)

    def expand(curr_node,white):
        if(len(curr_node.children)==0):
            return curr_node
        max_ucb = -inf
        if(white):
            idx = -1
            max_ucb = -inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i, level)
                if(tmp>max_ucb):
                    idx = i
                    max_ucb = tmp
                    sel_child = i

            return(expand(sel_child,0))

        else:
            idx = -1
            min_ucb = inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i, level)
                if(tmp<min_ucb):
                    idx = i
                    min_ucb = tmp
                    sel_child = i

            return expand(sel_child,1)

    def rollback(curr_node,reward):
        curr_node.n+=1
        curr_node.v+=reward
        while(curr_node.parent!=None):
            curr_node.N+=1
            curr_node = curr_node.parent
        return curr_node

    def mcts_pred(curr_node,over,white,iterations=10):
        if(over):
            return -1
        all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
        map_state_move = dict()
        
        for i in all_moves:
            tmp_state = chess.Board(curr_node.state.fen())
            tmp_state.push_san(i)
            child = node()
            child.state = tmp_state
            child.parent = curr_node
            curr_node.children.add(child)
            map_state_move[child] = i
            
        while(iterations>0):
            if(white):
                idx = -1
                max_ucb = -inf
                sel_child = None
                for i in curr_node.children:
                    tmp = ucb1(i, level)
                    if(tmp>max_ucb):
                        idx = i
                        max_ucb = tmp
                        sel_child = i
                ex_child = expand(sel_child,0)
                reward,state = rollout(ex_child)
                curr_node = rollback(state,reward)
                iterations-=1
            else:
                idx = -1
                min_ucb = inf
                sel_child = None
                for i in curr_node.children:
                    tmp = ucb1(i, level)
                    if(tmp<min_ucb):
                        idx = i
                        min_ucb = tmp
                        sel_child = i

                ex_child = expand(sel_child,1)

                reward,state = rollout(ex_child)

                curr_node = rollback(state,reward)
                iterations-=1
        if(white):
            
            mx = -inf
            idx = -1
            selected_move = ''
            for i in (curr_node.children):
                tmp = ucb1(i, level)
                if(tmp>mx):
                    mx = tmp
                    selected_move = map_state_move[i]
            return selected_move
        else:
            mn = inf
            idx = -1
            selected_move = ''
            for i in (curr_node.children):
                tmp = ucb1(i, level)
                if(tmp<mn):
                    mn = tmp
                    selected_move = map_state_move[i]
            return selected_move

    st.header("A Game of Chess using Monte Carlo Tree Search")
    board = chess.Board()
    # png = svg2png(bytestring=chess.svg.board(board))
    p1 = st.empty()
    imageLocation = st.empty()
    p3 = st.empty()
    p2 = st.empty()
    
    # Open png in PIL
    # pil_img = Image.open(BytesIO(png)).convert('RGBA')
    # cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    # imageLocation.image(cv_img, caption='Initial pose', width=600)
    svg = chess.svg.board(board)
    imageLocation.markdown(f"<div style='text-align: center'>{svg}</div>", unsafe_allow_html=True)

    white = 1
    moves = 0
    pgn = []
    game = chess.pgn.Game()
    evaluations = []
    sm = 0
    cnt = 0
    df = pd.DataFrame(columns=['Turn','Moves by White','Moves by Black'])
    iteration=1
    #gameboard=display.start(board.fen())
    if st.button("Start"):
        if(st.button("stop game")):
            st.stop()
        while((not board.is_game_over())):
            print("Iteration",iteration)
            iteration+=1
            l  = 'Total number of moves: '+str(moves)
            p3.subheader(l)
            all_moves = [board.san(i) for i in list(board.legal_moves)]
            #start = time.time()
            if(white):
                p1.subheader("White's Turn")
                root = node()
                root.state = board
                result = mcts_pred(root,board.is_game_over(),white)
                df.loc[len(df.index)] = [iteration-1, result, 'No move'] 
                #sm+=(time.time()-start)
                board.push_san(result)
            else:
                p1.subheader("Black's Turn")
                root = node()
                root.state = board
                result = mcts_pred(root,board.is_game_over(),white)
                df.loc[len(df.index)] = [iteration-1, 'No move',result ] 
                #sm+=(time.time()-start)
                board.push_san(result)

            #display.start(board.fen())
            #print(result)
            p2.table(df)
        #     png = svg2png(bytestring=chess.svg.board(board))
            
            
        # # Open png in PIL
        #     pil_img = Image.open(BytesIO(png)).convert('RGBA')
        #     cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
        #     imageLocation.image(cv_img, caption="Move "+result,width=600)
            svg = chess.svg.board(board)
            imageLocation.markdown(f"<div style='text-align: center'>{svg}</div>", unsafe_allow_html=True)


        # gameboard=display.update(board.fen())
            pgn.append(result)
   
            white ^= 1
            moves+=1
      
        #display.terminate()
            #board_evaluation = evaluate(board.fen().split()[0])
            #evaluations.append(board_evaluation)
        #print("Average Time per move = ",sm/cnt)



app()


# import streamlit as st
# from stchess import Board, DEFAULT_FEN
# import random
# import time
# from math import log,sqrt,e,inf
# import pandas as pd

# level = "easy"

# def app():
#     class node():
#         def __init__(self):
#             self.state = Board(DEFAULT_FEN)  # Sử dụng bàn cờ mặc định
#             self.action = ''
#             self.children = set()
#             self.parent = None
#             self.N = 0
#             self.n = 0
#             self.v = 0

#     def ucb1(curr_node, level):
#         if level == 'easy':
#             c = 2.5  
#         elif level == 'medium':
#             c = 1.4
#         else:
#             c = 0.8  
        
#         exploit = curr_node.v
#         explore = c * (sqrt(log(curr_node.N + e + (10**-6)) / (curr_node.n + (10**-10))))
#         return exploit + explore


#     def rollout(curr_node):
#         if curr_node.state.is_game_over():
#             if curr_node.state.result() == '1-0':
#                 return 1, curr_node
#             elif curr_node.state.result() == '0-1':
#                 return -1, curr_node
#             else:
#                 return 0.5, curr_node
        
#         all_moves = list(curr_node.state.legal_moves)
        
#         for move in all_moves:
#             tmp_state = Board(curr_node.state.fen())
#             tmp_state.push(move)
#             child = node()
#             child.state = tmp_state
#             child.parent = curr_node
#             curr_node.children.add(child)
        
#         rnd_state = random.choice(list(curr_node.children))
#         return rollout(rnd_state)

#     def expand(curr_node, white):
#         if len(curr_node.children) == 0:
#             return curr_node
#         max_ucb = -float('inf')
#         if white:
#             for i in curr_node.children:
#                 tmp = ucb1(i, level)
#                 if tmp > max_ucb:
#                     sel_child = i
#                     max_ucb = tmp
#             return expand(sel_child, 0)
#         else:
#             min_ucb = float('inf')
#             for i in curr_node.children:
#                 tmp = ucb1(i, level)
#                 if tmp < min_ucb:
#                     sel_child = i
#                     min_ucb = tmp
#             return expand(sel_child, 1)

#     def rollback(curr_node, reward):
#         curr_node.n += 1
#         curr_node.v += reward
#         while curr_node.parent != None:
#             curr_node.N += 1
#             curr_node = curr_node.parent
#         return curr_node

#     def mcts_pred(curr_node, over, white, iterations=10):
#         if over:
#             return -1
#         all_moves = list(curr_node.state.legal_moves)
#         map_state_move = dict()
        
#         for move in all_moves:
#             tmp_state = Board(curr_node.state.fen())
#             tmp_state.push(move)
#             child = node()
#             child.state = tmp_state
#             child.parent = curr_node
#             curr_node.children.add(child)
#             map_state_move[child] = move
            
#         while iterations > 0:
#             if white:
#                 max_ucb = -float('inf')
#                 for i in curr_node.children:
#                     tmp = ucb1(i, level)
#                     if tmp > max_ucb:
#                         sel_child = i
#                         max_ucb = tmp
#                 ex_child = expand(sel_child, 0)
#                 reward, state = rollout(ex_child)
#                 curr_node = rollback(state, reward)
#                 iterations -= 1
#             else:
#                 min_ucb = float('inf')
#                 for i in curr_node.children:
#                     tmp = ucb1(i, level)
#                     if tmp < min_ucb:
#                         sel_child = i
#                         min_ucb = tmp

#                 ex_child = expand(sel_child, 1)
#                 reward, state = rollout(ex_child)
#                 curr_node = rollback(state, reward)
#                 iterations -= 1
        
#         if white:
#             mx = -float('inf')
#             selected_move = ''
#             for i in curr_node.children:
#                 tmp = ucb1(i, level)
#                 if tmp > mx:
#                     mx = tmp
#                     selected_move = map_state_move[i]
#             return selected_move
#         else:
#             mn = float('inf')
#             selected_move = ''
#             for i in curr_node.children:
#                 tmp = ucb1(i, level)
#                 if tmp < mn:
#                     mn = tmp
#                     selected_move = map_state_move[i]
#             return selected_move

#     st.header("A Game of Chess using Monte Carlo Tree Search")
#     board = Board(DEFAULT_FEN)  # Bàn cờ ban đầu
#     p1 = st.empty()
#     imageLocation = st.empty()
#     p3 = st.empty()
#     p2 = st.empty()
    
#     svg = board.get_svg()
#     imageLocation.markdown(f"<div style='text-align: center'>{svg}</div>", unsafe_allow_html=True)

#     white = 1
#     moves = 0
#     pgn = []
#     game = []
#     evaluations = []
#     sm = 0
#     cnt = 0
#     df = pd.DataFrame(columns=['Turn','Moves by White','Moves by Black'])
#     iteration = 1

#     if st.button("Start"):
#         while not board.is_game_over():
#             print("Iteration", iteration)
#             iteration += 1
#             l = 'Total number of moves: ' + str(moves)
#             p3.subheader(l)
#             svg = board.get_svg()
#             imageLocation.markdown(f"<div style='text-align: center'>{svg}</div>", unsafe_allow_html=True)

#             if white:
#                 p1.subheader("White's Turn (Your move)")
#                 user_move = st.text_input("Enter your move in SAN format (e.g. e4, Nf3, Qxe5):", key=moves)
#                 if user_move and user_move in [str(m) for m in board.legal_moves]:
#                     board.push_san(user_move)
#                     df.loc[len(df.index)] = [iteration - 1, user_move, 'No move']
#                     white ^= 1
#                     moves += 1
#             else:
#                 p1.subheader("Black's Turn (Bot thinking...)")
#                 root = node()
#                 root.state = board
#                 result = mcts_pred(root, board.is_game_over(), white)
#                 board.push_san(result)
#                 df.loc[len(df.index)] = [iteration - 1, 'No move', result]
#                 white ^= 1
#                 moves += 1

#             p2.table(df)
#         st.success(f"Game over! Result: {board.result()}")

# app()
