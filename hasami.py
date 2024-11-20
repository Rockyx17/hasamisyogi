import numpy as np
import random
from collections import defaultdict

class HasamiShogi:
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=int)
        self.board[0] = 1  # 先手の駒を配置
        self.board[8] = 2  # 後手の駒を配置
        self.current_player = 1
        self.captured_pieces = {1: 0, 2: 0}  # 各プレイヤーが取った駒の数
        self.last_capture = 0  # 最後に駒を取ったプレイヤー

    def display_board_text(self):
        """テキストベースで盤面を表示"""
        print("  0 1 2 3 4 5 6 7 8")
        for i in range(9):
            row = f"{i} "
            for j in range(9):
                if self.board[i][j] == 0:
                    row += ". "
                elif self.board[i][j] == 1:
                    row += "○ "
                else:
                    row += "● "
            print(row)

    def _capture_pieces(self, x, y):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        total_captured = 0
        
        for dx, dy in directions:
            captured = self._check_capture(x, y, dx, dy)
            total_captured += captured
        
        if total_captured > 0:
            self.captured_pieces[self.current_player] += total_captured
            self.last_capture = self.current_player

    def _check_capture(self, x, y, dx, dy):
        opponent = 3 - self.current_player
        captured_positions = []
        current_x, current_y = x + dx, y + dy
        
        while (0 <= current_x < 9 and 0 <= current_y < 9 and 
               self.board[current_x][current_y] == opponent):
            captured_positions.append((current_x, current_y))
            current_x += dx
            current_y += dy
        
        if (0 <= current_x < 9 and 0 <= current_y < 9 and 
            self.board[current_x][current_y] == self.current_player):
            for pos_x, pos_y in captured_positions:
                self.board[pos_x][pos_y] = 0
            return len(captured_positions)
        
        return 0

    def is_game_over(self):
        if self.captured_pieces[1] >= 5 or self.captured_pieces[2] >= 5:
            return True
        
        diff = abs(self.captured_pieces[1] - self.captured_pieces[2])
        if diff >= 3:
            opponent = 3 - self.last_capture
            if self.current_player == opponent:
                for move in self.get_valid_moves():
                    test_game = self.clone()
                    test_game.make_move(*move)
                    if test_game.captured_pieces[opponent] > self.captured_pieces[opponent]:
                        return False
            return True
        
        return False

    def get_winner(self):
        if not self.is_game_over():
            return None
        
        if self.captured_pieces[1] >= 5:
            return 1
        elif self.captured_pieces[2] >= 5:
            return 2
        elif self.captured_pieces[1] - self.captured_pieces[2] >= 3:
            return 1
        elif self.captured_pieces[2] - self.captured_pieces[1] >= 3:
            return 2
        return None

    def clone(self):
        new_game = HasamiShogi()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.captured_pieces = self.captured_pieces.copy()
        new_game.last_capture = self.last_capture
        return new_game

    def get_valid_moves(self):
        valid_moves = []
        player_pieces = np.where(self.board == self.current_player)
    
        for piece_idx in range(len(player_pieces[0])):
            x, y = player_pieces[0][piece_idx], player_pieces[1][piece_idx]
            
            for new_x in range(9):
                if self._is_valid_move(x, y, new_x, y):
                    valid_moves.append((x, y, new_x, y))
            
            for new_y in range(9):
                if self._is_valid_move(x, y, x, new_y):
                    valid_moves.append((x, y, x, new_y))
    
        return valid_moves

    def _is_valid_move(self, from_x, from_y, to_x, to_y):
        if (from_x, from_y) == (to_x, to_y):
            return False
        
        if from_x != to_x and from_y != to_y:
            return False
        
        if self.board[from_x][from_y] != self.current_player:
            return False
        
        if self.board[to_x][to_y] != 0:
            return False
        
        if from_x == to_x:
            min_y = min(from_y, to_y)
            max_y = max(from_y, to_y)
            for y in range(min_y + 1, max_y):
                if self.board[from_x][y] != 0:
                    return False
        else:
            min_x = min(from_x, to_x)
            max_x = max(from_x, to_x)
            for x in range(min_x + 1, max_x):
                if self.board[x][from_y] != 0:
                    return False
        
        return True

    def make_move(self, from_x, from_y, to_x, to_y):
        if not self._is_valid_move(from_x, from_y, to_x, to_y):
            return False
        
        self.board[to_x][to_y] = self.current_player
        self.board[from_x][from_y] = 0
        
        self._capture_pieces(to_x, to_y)
        
        self.current_player = 3 - self.current_player
        
        return True

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.q_values = defaultdict(float)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, game):
        return (str(game.board.tolist()), 
                game.captured_pieces[1], 
                game.captured_pieces[2])

    def get_action(self, game):
        state = self.get_state_key(game)
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        best_value = float('-inf')
        best_actions = []
        
        for move in valid_moves:
            value = self.q_values[(state, str(move))]
            if value > best_value:
                best_value = value
                best_actions = [move]
            elif value == best_value:
                best_actions.append(move)
        
        return random.choice(best_actions)

    def get_max_q_value(self, game, state):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return 0
        return max([self.q_values[(state, str(move))] for move in valid_moves])

    def learn(self, game, state, action, reward, next_state):
        old_value = self.q_values[(state, str(action))]
        next_max = self.get_max_q_value(game, next_state)
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_values[(state, str(action))] = new_value

def train_ai(episodes=1000):
    agent = QLearningAgent()
    
    for episode in range(episodes):
        game = HasamiShogi()
        
        while not game.is_game_over():
            current_state = agent.get_state_key(game)
            action = agent.get_action(game)
            
            if action is None:
                break
            
            prev_captures = game.captured_pieces.copy()
            game.make_move(*action)
            
            reward = 0
            if game.is_game_over():
                winner = game.get_winner()
                if winner == 1:
                    reward = 10
                else:
                    reward = -10
            else:
                captures_diff = game.captured_pieces[1] - prev_captures[1]
                if captures_diff > 0:
                    reward = captures_diff
            
            new_state = agent.get_state_key(game)
            agent.learn(game, current_state, action, reward, new_state)
        
        if episode % 100 == 0:
            print(f"Episode {episode} completed")
    
    return agent

def play_against_ai(agent):
    # プレイヤーに先手/後手を選んでもらう
    while True:
        choice = input("先手(1)か後手(2)を選んでください: ")
        if choice in ['1', '2']:
            human_player = int(choice)
            break
        print("1か2を入力してください。")

    game = HasamiShogi()
    
    while not game.is_game_over():
        game.display_board_text()
        print(f"取った駒 - プレイヤー1: {game.captured_pieces[1]}, プレイヤー2: {game.captured_pieces[2]}")
        
        if game.current_player == human_player:
            valid_moves = game.get_valid_moves()
            print("有効な手:", valid_moves)
            
            while True:
                try:
                    from_x = int(input("移動元の行(0-8): "))
                    from_y = int(input("移動元の列(0-8): "))
                    to_x = int(input("移動先の行(0-8): "))
                    to_y = int(input("移動先の列(0-8): "))
                    
                    if game.make_move(from_x, from_y, to_x, to_y):
                        break
                    else:
                        print("無効な手です。もう一度入力してください。")
                except ValueError:
                    print("数値を入力してください。")
        else:
            action = agent.get_action(game)
            if action:
                print(f"AIの手: {action}")
                game.make_move(*action)
    
    game.display_board_text()
    winner = game.get_winner()
    print("ゲーム終了!")
    print(f"勝者: プレイヤー{winner}")
    print(f"最終スコア - プレイヤー1: {game.captured_pieces[1]}, プレイヤー2: {game.captured_pieces[2]}")

# AIを訓練して対戦
trained_agent = train_ai(1000)
play_against_ai(trained_agent)