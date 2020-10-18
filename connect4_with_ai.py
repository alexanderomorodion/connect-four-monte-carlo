import numpy as np
import random
import pygame
import sys
import math
import time
import threading

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

total_samples = 0
grand_total_samples = 0

payouts1 = {}
payouts2 = {}
payouts3 = {}
payouts4 = {}
grandPayouts = {}

threadLock = threading.Lock()
threads = []


class myThread (threading.Thread):
    def __init__(self, board, payouts):
        threading.Thread.__init__(self)
        self.board = board
        self.payouts = payouts

    def run(self):
        for i in range(4):
            monte_carlo(self.board, self.payouts)


def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flip(board, 0))


def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def calc_conf_interval(wins, draws, move_samples, total_samples):
    first_term = (wins + (draws / 2)) / move_samples
    second_term = math.sqrt(((np.log(total_samples)) / move_samples))
    return first_term + second_term


def monte_carlo(board, payouts):
    # dict of nodes and payouts
    count = 0
    start_time = time.time()
    seconds = 10
    global total_samples
    # check if position is terminal
    while not (is_terminal_node(board)):
        conf_interval = 0

        # begin monte carlo search
        for i in range(1000):
            board_copy = board.copy()
            selected_move = selection(board_copy, total_samples, payouts)
            total_samples += simulation(board_copy, selected_move, payouts)
            count += 1

        print(count)
        return selected_move


# calculates confidence interval for each valid move and returns move with highest interval.
# priority given to moves that are unsampled
def selection(board_copy, total_moves, payouts):
    max_conf = 0
    max_move = ()
    open_cols = get_valid_locations(board_copy)
    for col in open_cols:
        row = get_next_open_row(board_copy, col)
        node = (row, col)

        # check moves that haven't been tried yet
        if node not in payouts.keys():
            return node
        else:
            confidence_interval = calc_conf_interval(payouts[node][0], payouts[node][1], payouts[node][3], total_moves)
            if confidence_interval > max_conf:
                max_move = node
                max_conf = confidence_interval

    return max_move


# simulates random moves from starting move to determine winner
# updates wins/draws/losses/samples stats of each node simulated
# returns number of nodes simulated
def simulation(board_copy, move, payouts):
    sequence = []
    player_turn = 2
    total_moves = 1

    # move: (row, col)
    drop_piece(board_copy, move[0], move[1], player_turn)
    sequence.append(move)
    while not is_terminal_node(board_copy):
        # make a move
        valid_moves = get_valid_locations(board_copy)
        selected_move = random.choice(valid_moves)
        row = get_next_open_row(board_copy, selected_move)
        drop_piece(board_copy, row, selected_move, player_turn)
        total_moves += 1

        # add move to sequence
        node = (row, selected_move)
        sequence.append(node)

        # switch turns
        if player_turn == 2:
            player_turn = 1
        else:
            player_turn = 2

    # [wins, draws, losses, total_trials]
    # backpropagation
    if winning_move(board_copy, PLAYER_PIECE):
        for node in sequence:
            if node not in payouts:
                payouts[node] = [0, 0, 1, 1]
            else:
                # add 1 to losses and total trials
                payouts[node][2] += 1
                payouts[node][3] += 1

    elif winning_move(board_copy, AI_PIECE):
        for node in sequence:
            if node not in payouts:
                payouts[node] = [1, 0, 0, 1]
            else:
                # add 1 to wins and total trials
                payouts[node][0] += 1
                payouts[node][3] += 1
    else:
        for node in sequence:
            if node not in payouts:
                payouts[node] = [0, 1, 0, 1]
            else:
                # add 1 to draws and total trials
                payouts[node][1] += 1
                payouts[node][3] += 1
    return total_moves


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
                int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                    int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


def join_payouts():
    global grand_total_samples
    global grandPayouts
    for node in payouts1.keys():
        if node not in grandPayouts:
            grandPayouts[node] = payouts1[node]
        else:
            grandPayouts[node][0] += payouts1[node][0]
            grandPayouts[node][1] += payouts1[node][1]
            grandPayouts[node][2] += payouts1[node][2]
            grandPayouts[node][3] += payouts1[node][3]
        if node in payouts2.keys():
            grandPayouts[node][0] += payouts2[node][0]
            grandPayouts[node][1] += payouts2[node][1]
            grandPayouts[node][2] += payouts2[node][2]
            grandPayouts[node][3] += payouts2[node][3]
        if node in payouts3.keys():
            grandPayouts[node][0] += payouts3[node][0]
            grandPayouts[node][1] += payouts3[node][1]
            grandPayouts[node][2] += payouts3[node][2]
            grandPayouts[node][3] += payouts3[node][3]
        if node in payouts4.keys():
            grandPayouts[node][0] += payouts4[node][0]
            grandPayouts[node][1] += payouts4[node][1]
            grandPayouts[node][2] += payouts4[node][2]
            grandPayouts[node][3] += payouts4[node][3]
        grand_total_samples += grandPayouts[node][3]
    return grand_total_samples


if __name__ == '__main__':
    board = create_board()
    print_board(board)
    game_over = False

    pygame.init()

    SQUARESIZE = 100

    width = COLUMN_COUNT * SQUARESIZE
    height = (ROW_COUNT + 1) * SQUARESIZE

    size = (width, height)

    RADIUS = int(SQUARESIZE / 2 - 5)

    screen = pygame.display.set_mode(size)
    draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    turn = random.randint(PLAYER, AI)
    print(turn)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                if turn == PLAYER:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                # print(event.pos)
                # Ask for Player 1 Input
                if turn == PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, PLAYER_PIECE)

                        if winning_move(board, PLAYER_PIECE):
                            label = myfont.render("Player 1 wins!!", 1, RED)
                            screen.blit(label, (40, 10))
                            game_over = True

                        turn += 1
                        turn = turn % 2

                        print_board(board)
                        draw_board(board)

        # # Ask for Player 2 Input
        if turn == AI and not game_over:
            # Create new threads
            thread1 = myThread(board, payouts1)
            thread2 = myThread(board, payouts2)
            thread3 = myThread(board, payouts3)
            thread4 = myThread(board, payouts4)

            # Start new threads
            thread1.start()
            thread2.start()
            thread3.start()
            thread4.start()

            # Add threads to thread list
            threads.append(thread1)
            threads.append(thread2)
            threads.append(thread3)
            threads.append(thread4)

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Perform one more selection based on grand total of payouts
            grandTotal = join_payouts()
            AI_move = selection(board, grandTotal, grandPayouts)
            drop_piece(board, AI_move[0], AI_move[1], AI_PIECE)
            if winning_move(board, AI_PIECE):
                label = myfont.render("Player 2 wins!!", 1, YELLOW)
                screen.blit(label, (40, 10))
                game_over = True

            print_board(board)
            draw_board(board)

            turn += 1
            turn = turn % 2

        if game_over:
            pygame.time.wait(3000)
