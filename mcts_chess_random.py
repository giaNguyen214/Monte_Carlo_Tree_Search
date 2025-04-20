import chess  # Chess library
import tkinter as tk
from PIL import Image, ImageTk
import os
import math
import random

# === MCTS Logic ===
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = list(board.legal_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=0.5):
        # Avoid division by zero
        return max(self.children, key=lambda child:
            (child.wins / child.visits if child.visits > 0 else 0) + 
            c_param * math.sqrt(math.log(self.visits) / child.visits if child.visits > 0 else 0)
        ) if self.children else None

    def expand(self):
        move = self.untried_moves.pop()
        next_board = self.board.copy()
        next_board.push(move)
        child = MCTSNode(next_board, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

def simulate_random_game(board):
    board = board.copy()
    while not board.is_game_over():
        moves = list(board.legal_moves)
        if not moves:  # Safety check
            break
        move = random.choice(moves)
        board.push(move)
    
    # Determine result
    if board.is_checkmate():
        return 1 if not board.turn else 0  # If it's black's turn, white won
    return 0.5  # Draw

def mcts(board, iter_limit=1000, c_param=1.4):
    root = MCTSNode(board)
    
    for _ in range(iter_limit):
        # Selection
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(c_param)
            if node is None:  # Safety check
                break
                
        # Expansion
        if node and not node.is_fully_expanded() and not node.board.is_game_over():
            node = node.expand()
            
        # Simulation
        if node:
            result = simulate_random_game(node.board)
            
            # Backpropagation
            while node:
                node.update(result if node.board.turn == board.turn else 1 - result)
                node = node.parent
    
    # Choose the move with the most visits
    if not root.children:
        # If no children (shouldn't happen in normal play), pick a random move
        return random.choice(list(board.legal_moves))
        
    return max(root.children, key=lambda n: n.visits).move

# === Chess pieces representation with Unicode characters ===
def get_piece_symbol(piece):
    """Convert chess.Piece to Unicode character"""
    if not piece:
        return " "
    
    symbols = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }
    return symbols[piece.symbol()]

# === GUI ===
class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess with MCTS AI")
        self.board = chess.Board()
        self.square_size = 160  # Smaller squares to fit settings panel
        
        # Game state
        self.player_color = chess.WHITE  # Default player color
        self.difficulty_level = "Medium"  # Default difficulty
        self.selected_square = None
        self.difficulty_iterations = {
            "Easy": 100,
            "Medium": 500,
            "Hard": 1000
        }
        
        # Main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Settings frame (left side)
        self.settings_frame = tk.Frame(self.main_frame, width=200, bg="#f0f0f0", padx=10, pady=10)
        self.settings_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Color selection
        self.color_label = tk.Label(self.settings_frame, text="Play as:", bg="#f0f0f0")
        self.color_label.pack(pady=(0, 5), anchor="w")
        
        self.color_var = tk.StringVar(value="White")
        self.white_radio = tk.Radiobutton(self.settings_frame, text="White", variable=self.color_var, 
                                        value="White", command=self.update_player_color, bg="#f0f0f0")
        self.white_radio.pack(anchor="w")
        
        self.black_radio = tk.Radiobutton(self.settings_frame, text="Black", variable=self.color_var, 
                                        value="Black", command=self.update_player_color, bg="#f0f0f0")
        self.black_radio.pack(anchor="w", pady=(0, 10))
        
        # Difficulty selection
        self.diff_label = tk.Label(self.settings_frame, text="Difficulty:", bg="#f0f0f0")
        self.diff_label.pack(pady=(10, 5), anchor="w")
        
        self.diff_var = tk.StringVar(value="Medium")
        self.easy_radio = tk.Radiobutton(self.settings_frame, text="Easy", variable=self.diff_var, 
                                        value="Easy", command=self.update_difficulty, bg="#f0f0f0")
        self.easy_radio.pack(anchor="w")
        
        self.medium_radio = tk.Radiobutton(self.settings_frame, text="Medium", variable=self.diff_var, 
                                        value="Medium", command=self.update_difficulty, bg="#f0f0f0")
        self.medium_radio.pack(anchor="w")
        
        self.hard_radio = tk.Radiobutton(self.settings_frame, text="Hard", variable=self.diff_var, 
                                        value="Hard", command=self.update_difficulty, bg="#f0f0f0")
        self.hard_radio.pack(anchor="w", pady=(0, 10))
        
        # New game button
        self.new_game_button = tk.Button(self.settings_frame, text="New Game", command=self.new_game)
        self.new_game_button.pack(pady=10, fill=tk.X)
        
        # Status label
        self.status_label = tk.Label(self.settings_frame, text="Game in progress", bg="#f0f0f0", 
                                     wraplength=180, justify=tk.LEFT)
        self.status_label.pack(pady=10, anchor="w")
        
        # Game frame (right side)
        self.game_frame = tk.Frame(self.main_frame)
        self.game_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas for the chess board
        self.canvas = tk.Canvas(self.game_frame, width=8*self.square_size, height=8*self.square_size)
        self.canvas.pack(padx=20, pady=20)
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.draw_board()
        
        # If AI goes first (player is black)
        if self.board.turn != self.player_color:
            self.root.after(100, self.ai_move)

    def update_player_color(self):
        """Update player color based on radio button selection"""
        new_color = chess.WHITE if self.color_var.get() == "White" else chess.BLACK
        
        # Only restart if color actually changed
        if new_color != self.player_color:
            self.player_color = new_color
            self.new_game()  # Restart the game with new settings

    def update_difficulty(self):
        """Update difficulty level based on radio button selection"""
        self.difficulty_level = self.diff_var.get()
        self.status_label.config(text=f"Difficulty set to {self.difficulty_level}")

    def new_game(self):
        """Reset the game board"""
        self.board = chess.Board()
        self.selected_square = None
        self.draw_board()
        self.update_status()
        
        # If AI goes first
        if self.board.turn != self.player_color:
            self.root.after(100, self.ai_move)

    def update_status(self):
        """Update the status label based on game state"""
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn == chess.WHITE else "White"
                self.status_label.config(text=f"Checkmate! {winner} wins")
            else:
                self.status_label.config(text="Game over: Draw")
        else:
            turn = "Your" if self.board.turn == self.player_color else "AI's"
            self.status_label.config(text=f"{turn} turn\nPlaying as: {'White' if self.player_color == chess.WHITE else 'Black'}\nDifficulty: {self.difficulty_level}")

    def draw_board(self):
        self.canvas.delete("all")
        colors = ["#F0D9B5", "#B58863"]  # Light, Dark
        
        # Draw squares
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = (7 - rank) * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = colors[(rank + file) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
        # Highlight selected square if any
        if self.selected_square is not None:
            file = chess.square_file(self.selected_square)
            rank = chess.square_rank(self.selected_square)
            x1 = file * self.square_size
            y1 = (7 - rank) * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ff0000", width=3)
        
        # Draw pieces
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece:
                    x = file * self.square_size + self.square_size // 2
                    y = (7 - rank) * self.square_size + self.square_size // 2
                    symbol = get_piece_symbol(piece)
                    color = "black" if piece.color == chess.BLACK else "white"
                    # Use large font for piece symbols
                    self.canvas.create_text(x, y, text=symbol, font=("Arial", 36), fill=color)
        
        self.update_status()

    def on_click(self, event):
        if self.board.is_game_over() or self.board.turn != self.player_color:
            return
            
        file = event.x // self.square_size
        rank = 7 - event.y // self.square_size
        square = chess.square(file, rank)
        
        if self.selected_square is None:
            # Select piece if it's the player's
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
                self.draw_board()
        else:
            # Try to move the selected piece
            move = chess.Move(self.selected_square, square)
            
            # Check for promotion
            if (self.board.piece_at(self.selected_square) and 
                self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                ((self.player_color == chess.WHITE and rank == 7) or
                 (self.player_color == chess.BLACK and rank == 0))):
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            # Make the move if legal
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                
                # AI responds after a short delay
                if not self.board.is_game_over():
                    self.status_label.config(text="AI thinking...")
                    self.root.update()
                    self.root.after(100, self.ai_move)
            else:
                # If clicked on own piece, select that instead
                piece = self.board.piece_at(square)
                if piece and piece.color == self.player_color:
                    self.selected_square = square
                else:
                    self.selected_square = None
                self.draw_board()

    def ai_move(self):
        if self.board.is_game_over():
            self.update_status()
            return
            
        # Update status to show AI is thinking
        self.status_label.config(text="AI thinking...")
        self.root.update()
        
        # Use MCTS with iterations based on difficulty
        iterations = self.difficulty_iterations[self.difficulty_level]
        move = mcts(self.board, iter_limit=iterations)
        self.board.push(move)
        self.draw_board()

# === Run GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()