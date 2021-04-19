# RL-Tag-game
An attempt at making a RL model for a simple game of tag

To run: Clone repo, run main.py. Click on areas of the pop-up window to place walls, re-click on existing walls to remove. Press ESC to quit, or press ENTER to proceed with that wall formation (List representing walls are printed in console). You will be prompted to enter number of iterations of the game in the console, type an integer larger than 0 to start. If seeker catches runner, appropriate text output in console. Press ESC in the game window to force stop the current game.

#Variables that can be changed and their locations.
tag,walls - window size.
tag - seeker_move_mode, runner_move_mode, entity size (seeker_size, runner_size), momentum constant (acceleration), entity colours (when making instance seeker, runner), fps (fps)
walls - dimension of each "wall" block (wall_size)
