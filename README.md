Chess with MCTS AI
Overview
This project implements a chess game with a graphical user interface (GUI) using Python's tkinter library and the python-chess library for chess logic. The AI opponent uses the Monte Carlo Tree Search (MCTS) algorithm to make moves, with adjustable difficulty levels.
Features

Play as White or Black against an AI opponent.
Three difficulty levels: Easy, Medium, and Hard, adjusting the MCTS exploration parameter.
Interactive chessboard with piece movement via mouse clicks.
Support for pawn promotion (automatically promotes to Queen).
Game status updates, including checkmate and draw detection.
Option to start a new game at any time.
Unicode chess piece symbols for a clean visual representation.

Requirements

Python 3.6+
Required Python libraries:
python-chess (for chess logic and board management)
tkinter (for the GUI, usually included with Python)
Pillow (for image handling, though not used in this version)


The GUI will open, allowing you to:

Choose to play as White or Black.
Select a difficulty level (Easy, Medium, Hard).
Click on squares to move pieces (click a piece to select, then click a destination).
Click "New Game" to reset the board.



Project Structure

MCTS logic for the AI (MCTSNode class and related functions).
Chess piece representation using Unicode symbols.
GUI implementation using tkinter (ChessGUI class).


README.md: This file, providing project documentation.

How the AI Works
The AI uses the Monte Carlo Tree Search (MCTS) algorithm, which:

Selection: Traverses the game tree to find a promising node.
Expansion: Adds a new move to the tree if untried moves exist.
Simulation: Plays a random game from the current position to estimate the move's value.
Backpropagation: Updates win/visit statistics for all nodes in the path.

The difficulty level adjusts the exploration parameter (c_param):

Easy: c_param = 0.1 (more exploitative, less exploration).
Medium: c_param = 0.5 (balanced).
Hard: c_param = 1.0 (more exploration, stronger play).

Usage Notes

The board is oriented with White at the bottom (rank 1) and Black at the top (rank 8).
When playing as Black, the AI (White) moves first automatically.
Pawn promotion is automatic (to Queen); future versions could add a selection dialog.
The AI may take a moment to "think" (especially on Hard difficulty) due to the MCTS iterations (1000 by default).
The game detects checkmate and stalemate, updating the status label accordingly.

Future Improvements

Add a dialog for pawn promotion to choose piece types.
Implement move history or undo functionality.
Enhance AI performance by increasing MCTS iterations or adding heuristics.
Add visual highlights for legal moves when a piece is selected.
Include sound effects or animations for moves.
Support for saving/loading games.

License
This project is open-source and available under the MIT License.
Acknowledgments

The python-chess library for robust chess mechanics.
The MCTS algorithm, inspired by its applications in game AI.

For any issues or contributions, please open an issue or pull request on the project repository (if hosted).
