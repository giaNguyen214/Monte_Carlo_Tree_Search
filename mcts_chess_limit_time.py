import chess  # Chess library
import tkinter as tk
from tkinter import ttk
import math
import random
import threading
import time
from functools import lru_cache

# === Enhanced MCTS Logic with optimizations ===
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = list(board.legal_moves)
        # Prioritize checking, capturing and center control moves
        self.prioritize_moves()
        
    def prioritize_moves(self):
        """Sort moves to check promising ones first"""
        scored_moves = []
        for move in self.untried_moves:
            score = 0
            # Check if move gives check
            board_copy = self.board.copy()
            board_copy.push(move)
            if board_copy.is_check():
                score += 10
                
            # Check if move is a capture
            if self.board.is_capture(move):
                # Value captures by piece value
                victim_value = get_piece_value(self.board.piece_at(move.to_square))
                attacker_value = get_piece_value(self.board.piece_at(move.from_square))
                score += victim_value - attacker_value/10
                
            # Prioritize center control in opening/middlegame
            if not is_endgame(self.board):
                # Center squares
                center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
                if move.to_square in center_squares:
                    score += 3
            
            scored_moves.append((move, score))
            
        # Sort by score (descending)
        scored_moves.sort(key=lambda x: x[1], reverse=True)  
        self.untried_moves = [move for move, _ in scored_moves]

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        if not self.children:
            return None
            
        # UCB1 formula with improvements
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            # Avoid division by zero
            if child.visits == 0:
                continue
                
            # Win rate component
            win_rate = child.wins / child.visits
            
            # Exploration component
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            
            # Combined score
            score = win_rate + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def expand(self):
        if not self.untried_moves:
            return None
            
        move = self.untried_moves.pop(0)  # Get first (highest priority) move
        next_board = self.board.copy()
        next_board.push(move)
        child = MCTSNode(next_board, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

def get_piece_value(piece):
    """Get standard piece values for evaluation"""
    if not piece:
        return 0
        
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3, 
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100
    }
    return piece_values[piece.piece_type]

def is_endgame(board):
    """Check if position is likely in endgame phase"""
    # Count material
    total_material = sum(get_piece_value(p) for p in board.piece_map().values() 
                      if p.piece_type != chess.KING)
    # Endgame is when material is below threshold or no queens
    return total_material < 15 or not any(p.piece_type == chess.QUEEN for p in board.piece_map().values())

@lru_cache(maxsize=10000)
def evaluate_position(board_fen):
    """Evaluate a chess position (higher values favor white)"""
    board = chess.Board(board_fen)
    
    if board.is_checkmate():
        return -1000 if board.turn else 1000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
        
    # Material counting
    material_score = 0
    pieces = board.piece_map()
    
    for square, piece in pieces.items():
        value = get_piece_value(piece)
        # Add position-based adjustments
        if piece.piece_type == chess.PAWN:
            # Bonus for advanced pawns
            rank = chess.square_rank(square)
            bonus = rank if piece.color == chess.WHITE else 7 - rank
            value += bonus * 0.1
            
        # Apply color factor
        factor = 1 if piece.color == chess.WHITE else -1
        material_score += factor * value
    
    # Check threats
    if board.is_check():
        if board.turn == chess.WHITE:
            material_score -= 0.5  # White is in check
        else:
            material_score += 0.5  # Black is in check
            
    # Mobility (number of legal moves)
    mobility = len(list(board.legal_moves))
    # Switch turn to calculate opponent mobility
    board_copy = board.copy()
    board_copy.push(chess.Move.null())
    opponent_mobility = len(list(board_copy.legal_moves))
    mobility_score = 0.05 * (mobility - opponent_mobility)
    
    if board.turn == chess.WHITE:
        material_score += mobility_score
    else:
        material_score -= mobility_score
        
    return material_score

def simulate_smart_game(board):
    """Smarter simulation that uses basic evaluation"""
    board = board.copy()
    moves = 0
    max_moves = 100  # Prevent infinite games
    
    while not board.is_game_over() and moves < max_moves:
        # If less than 3 legal moves, evaluate all
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
            
        if len(legal_moves) <= 3 or random.random() < 0.7:  # 70% chance of making a smart move
            # Smart move selection
            best_score = float('-inf')
            best_move = None
            
            for move in legal_moves:
                board_copy = board.copy()
                board_copy.push(move)
                score = evaluate_position(board_copy.fen())
                # Adjust score for perspective
                if board.turn == chess.BLACK:
                    score = -score
                    
                if score > best_score:
                    best_score = score
                    best_move = move
            
            board.push(best_move)
        else:
            # Sometimes play randomly to add variety
            move = random.choice(legal_moves)
            board.push(move)
            
        moves += 1
    
    # Determine result
    if board.is_checkmate():
        return 1 if not board.turn else 0  # If it's black's turn, white won
    return 0.5  # Draw

def mcts(board, iter_limit=1000, time_limit=5.0, c_param=1.4):
    root = MCTSNode(board)
    start_time = time.time()
    iterations = 0
    
    # Use either iteration limit or time limit
    while iterations < iter_limit and (time.time() - start_time) < time_limit:
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
            result = simulate_smart_game(node.board)
            
            # Backpropagation
            while node:
                node.update(result if node.board.turn == board.turn else 1 - result)
                node = node.parent
                
        iterations += 1
    
    # Choose the best move
    if not root.children:
        # If no children (shouldn't happen in normal play), pick a random move
        return random.choice(list(board.legal_moves))
        
    # Return move with best win rate for serious play
    best_child = max(root.children, key=lambda n: n.wins/n.visits if n.visits > 0 else 0)
    return best_child.move

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
        self.root.title("Chess with Enhanced MCTS AI")
        self.board = chess.Board()
        self.square_size = 160  # Smaller squares to fit settings panel
        
        # Game state
        self.player_color = chess.WHITE  # Default player color
        self.difficulty_level = "Medium"  # Default difficulty
        self.ai_speed = 50  # Default speed percentage (higher = faster)
        self.selected_square = None
        self.ai_thinking = False  # Flag to track if AI is currently thinking
        
        # AI parameters
        self.difficulty_iterations = {
            "Easy": 100,
            "Medium": 500,
            "Hard": 1500
        }
        self.difficulty_time_limits = {
            "Easy": 0.5,  # seconds
            "Medium": 1.5,
            "Hard": 3.0
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
        
        # AI Speed slider
        self.speed_label = tk.Label(self.settings_frame, text="AI Speed:", bg="#f0f0f0")
        self.speed_label.pack(pady=(10, 5), anchor="w")
        
        self.speed_var = tk.IntVar(value=50)
        self.speed_slider = ttk.Scale(self.settings_frame, from_=1, to=100, 
                                    variable=self.speed_var, orient=tk.HORIZONTAL,
                                    command=self.update_speed_label)
        self.speed_slider.pack(fill=tk.X)
        
        self.speed_value_label = tk.Label(self.settings_frame, text="50%", bg="#f0f0f0")
        self.speed_value_label.pack(pady=(0, 10))
        
        # New game button
        self.new_game_button = tk.Button(self.settings_frame, text="New Game", command=self.new_game)
        self.new_game_button.pack(pady=10, fill=tk.X)

        # Undo move button
        self.undo_button = tk.Button(self.settings_frame, text="Undo Move", command=self.undo_move)
        self.undo_button.pack(pady=10, fill=tk.X)
        
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
        
        # AI stats
        self.ai_stats_label = tk.Label(self.settings_frame, text="AI Stats:", bg="#f0f0f0",
                                      wraplength=180, justify=tk.LEFT)
        self.ai_stats_label.pack(pady=(20, 5), anchor="w")
        
        self.draw_board()
        
        # If AI goes first (player is black)
        if self.board.turn != self.player_color:
            self.ai_move()

    def update_speed_label(self, event=None):
        """Update the label showing the AI speed percentage"""
        speed = self.speed_var.get()
        self.speed_value_label.config(text=f"{speed}%")
        self.ai_speed = speed

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
        # If AI is thinking, we need to wait for it to finish
        if self.ai_thinking:
            self.status_label.config(text="Please wait for AI to finish move before starting new game")
            return
            
        self.board = chess.Board()
        self.selected_square = None
        self.draw_board()
        self.update_status()
        
        # If AI goes first
        if self.board.turn != self.player_color:
            self.ai_move()

    def undo_move(self):
        """Undo the last two moves (player and AI)"""
        if self.ai_thinking:
            self.status_label.config(text="Cannot undo while AI is thinking")
            return
            
        # Pop two moves if possible (player's move and AI's response)
        moves_to_undo = 2
        while moves_to_undo > 0 and len(self.board.move_stack) > 0:
            self.board.pop()
            moves_to_undo -= 1
            
        self.selected_square = None
        self.draw_board()

    def update_status(self):
        """Update the status label based on game state"""
        if self.board.is_game_over():
            if self.board.is_checkmate():
                winner = "Black" if self.board.turn == chess.WHITE else "White"
                self.status_label.config(text=f"Checkmate! {winner} wins")
            else:
                result = "Draw"
                if self.board.is_stalemate():
                    result += " (Stalemate)"
                elif self.board.is_insufficient_material():
                    result += " (Insufficient material)"
                elif self.board.is_seventyfive_moves():
                    result += " (75-move rule)"
                elif self.board.is_fivefold_repetition():
                    result += " (Repetition)"
                self.status_label.config(text=f"Game over: {result}")
        else:
            turn = "Your" if self.board.turn == self.player_color else "AI's"
            self.status_label.config(text=f"{turn} turn\n"
                                     f"Playing as: {'White' if self.player_color == chess.WHITE else 'Black'}\n"
                                     f"Difficulty: {self.difficulty_level}\n"
                                     f"AI Speed: {self.ai_speed}%")

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
            
            # Highlight possible moves for selected piece
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    dest_file = chess.square_file(move.to_square)
                    dest_rank = chess.square_rank(move.to_square)
                    cx = dest_file * self.square_size + self.square_size // 2
                    cy = (7 - dest_rank) * self.square_size + self.square_size // 2
                    
                    # Different highlight for captures
                    if self.board.is_capture(move):
                        self.canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, 
                                               outline="#ff0000", width=2)
                    else:
                        self.canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, 
                                               fill="#00aa00")
        
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
        
        # Draw board coordinates
        for i in range(8):
            # Files (columns - a through h)
            x = i * self.square_size + self.square_size // 2
            y = 8 * self.square_size - 8
            self.canvas.create_text(x, y, text=chr(97 + i), font=("Arial", 10))
            
            # Ranks (rows - 1 through 8)
            x = 2
            y = (7 - i) * self.square_size + self.square_size // 2
            self.canvas.create_text(x, y, text=str(i + 1), font=("Arial", 10))
        
        self.update_status()
        
        # Display material advantage
        evaluation = evaluate_position(self.board.fen())
        if abs(evaluation) < 0.1:
            eval_text = "Equal position"
        else:
            side = "White" if evaluation > 0 else "Black"
            advantage = abs(evaluation)
            if advantage > 900:  # Checkmate
                eval_text = f"{side} is winning (Checkmate)"
            else:
                eval_text = f"{side} advantage: +{advantage:.1f}"
                
        self.ai_stats_label.config(text=f"Position evaluation:\n{eval_text}")

    def on_click(self, event):
        if self.board.is_game_over() or self.board.turn != self.player_color or self.ai_thinking:
            return
            
        file = event.x // self.square_size
        rank = 7 - event.y // self.square_size
        
        # Check if click is inside the board
        if file < 0 or file > 7 or rank < 0 or rank > 7:
            return
            
        square = chess.square(file, rank)
        
        if self.selected_square is None:
            # Select piece if it's the player's
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
                self.draw_board()
        else:
            # Check if clicked on a valid destination for selected piece
            move = chess.Move(self.selected_square, square)
            
            # Check for promotion
            if (self.board.piece_at(self.selected_square) and 
                self.board.piece_at(self.selected_square).piece_type == chess.PAWN and
                ((self.player_color == chess.WHITE and rank == 7) or
                 (self.player_color == chess.BLACK and rank == 0))):
                # Show promotion dialog
                self.show_promotion_dialog(self.selected_square, square)
                return
            
            # Make the move if legal
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                
                # AI responds after a short delay
                if not self.board.is_game_over():
                    self.status_label.config(text="AI thinking...")
                    self.root.update()
                    self.ai_move()
            else:
                # If clicked on own piece, select that instead
                piece = self.board.piece_at(square)
                if piece and piece.color == self.player_color:
                    self.selected_square = square
                else:
                    self.selected_square = None
                self.draw_board()
    
    def show_promotion_dialog(self, from_square, to_square):
        """Show dialog for pawn promotion piece selection"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Promote Pawn")
        dialog.geometry("250x60")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        
        # Promotion options
        options_frame = tk.Frame(dialog)
        options_frame.pack(fill=tk.BOTH, expand=True)
        
        # Pieces to choose from
        pieces = ["Queen", "Rook", "Bishop", "Knight"]
        piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        
        for i, (piece_name, piece_type) in enumerate(zip(pieces, piece_types)):
            button = tk.Button(options_frame, text=piece_name, width=6,
                              command=lambda pt=piece_type: self.handle_promotion(from_square, to_square, pt, dialog))
            button.grid(row=0, column=i, padx=5, pady=5)
            
        # Center dialog over main window
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Make modal
        dialog.grab_set()
        self.root.wait_window(dialog)
        
    def handle_promotion(self, from_square, to_square, promotion_piece, dialog):
        """Handle pawn promotion choice"""
        # Create and push promotion move
        move = chess.Move(from_square, to_square, promotion=promotion_piece)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.selected_square = None
            self.draw_board()
            
            # Close dialog
            dialog.destroy()
            
            # AI responds
            if not self.board.is_game_over():
                self.status_label.config(text="AI thinking...")
                self.root.update()
                self.ai_move()
    
    def ai_move(self):
        """Handle AI move in a separate thread to keep UI responsive"""
        if self.board.is_game_over():
            self.update_status()
            return
        
        # Set flag that AI is thinking
        self.ai_thinking = True
        self.status_label.config(text="AI thinking...")
        self.root.update()
        
        # Calculate how many iterations and time based on difficulty and speed
        base_iterations = self.difficulty_iterations[self.difficulty_level]
        base_time = self.difficulty_time_limits[self.difficulty_level]
        
        speed_factor = self.ai_speed / 100  # Convert percentage to factor
        
        # Adjust parameters based on speed
        adjusted_iterations = max(int(base_iterations * (1 - speed_factor * 0.9)), 10)
        adjusted_time = max(base_time * (1 - speed_factor * 0.9), 0.1)  # Minimum time 0.1s
        
        # Start AI thinking in a separate thread
        threading.Thread(target=self.process_ai_move, 
                        args=(adjusted_iterations, adjusted_time)).start()
    
    def process_ai_move(self, iterations, time_limit):
        """Process AI move in background thread"""
        start_time = time.time()
        
        # Use enhanced MCTS with time and iteration limits
        move = mcts(self.board, iter_limit=iterations, time_limit=time_limit)
        
        # Calculate thinking time
        thinking_time = time.time() - start_time
        
        # Schedule the move to be applied in the main thread
        self.root.after(0, self.apply_ai_move, move, thinking_time)
    
    def apply_ai_move(self, move, thinking_time):
        """Apply the AI's move on the main thread"""
        if not self.board.is_game_over():
            # Show thinking time
            self.ai_stats_label.config(text=f"AI thinking time: {thinking_time:.2f}s")
            
            # Make the move
            self.board.push(move)
            self.draw_board()
        
        # Reset AI thinking flag
        self.ai_thinking = False
        self.update_status()

# === Run GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()