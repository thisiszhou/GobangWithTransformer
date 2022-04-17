from game.cons import GAME_PLAYER
from game.game import Game
import numpy as np
from loguru import logger
from collections import defaultdict
import time
from typing import Dict, Optional, Callable, Union
import pygame


class VisualChessConfig:
    REC_SIZE = 50  # 棋子移动空间的大小
    CHESS_RADIUS = REC_SIZE // 2 - 2
    INFO_WIDTH = 200  # 信息宽度
    BUTTON_WIDTH = 140  # 按钮宽度
    BUTTON_HEIGHT = 50  # 按钮高度
    SEARCH_DEPTH = 5  # 搜索深度5
    LIMITED_MOVE_NUM = 10  # 限制步数10
    BG_MAIN_FILE = "picture/bgmain.jpg"
    BG_SIDE_FILE = "picture/bgbian.jpg"
    PLAY1_WIN_FILE = "picture/onewin.jpg"
    PLAY2_WIN_FILE = "picture/twowin.jpg"

    def __init__(self, chess_size):
        self.CHESS_LEN = chess_size
        self.MAP_WIDTH = self.CHESS_LEN * self.REC_SIZE  # 地图宽度
        self.MAP_HEIGHT = self.CHESS_LEN * self.REC_SIZE  # 地图高度
        self.SCREEN_WIDTH = self.MAP_WIDTH + self.INFO_WIDTH  # 屏幕宽度
        self.SCREEN_HEIGHT = self.MAP_HEIGHT  # 屏幕高度


class Displayer(object):
    def __init__(self, conf, caption="Gobang"):
        self.conf = conf
        pygame.init()
        self.screen = pygame.display.set_mode([self.conf.SCREEN_WIDTH, self.conf.SCREEN_HEIGHT])
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.buttons = {
            'start': StartButton(self.screen, 'start', self.conf.MAP_WIDTH + 30, 15, self.conf),
            'surrend': GiveupButton(self.screen, 'surrend', self.conf.MAP_WIDTH + 30,
                                    self.conf.BUTTON_HEIGHT + 45, self.conf),
            'change': ChangeUserButton(self.screen, 'change', self.conf.MAP_WIDTH + 30,
                                       2 * self.conf.BUTTON_HEIGHT + 75, self.conf)
        }
        # background
        self.bg_main = pygame.image.load(self.conf.BG_MAIN_FILE)
        self.bg_side = pygame.image.load(self.conf.BG_SIDE_FILE)
        self.play1_win = pygame.image.load(self.conf.PLAY1_WIN_FILE)
        self.play2_win = pygame.image.load(self.conf.PLAY2_WIN_FILE)

    def show_fps(self, steps, board, empty_method, fps=100, winner=None):
        self.clock.tick(fps)
        self.screen.blit(self.bg_main, (0, 0))
        self.screen.blit(self.bg_side, (self.conf.MAP_WIDTH, 0))
        for button in self.buttons.values():
            button.draw()
        self.change_mouse_show(empty_method)
        if winner is not None:
            self.show_winner(winner)
        self.draw_background()
        self.draw_chess(steps, board)
        pygame.display.update()

    def show_winner(self, winner):
        def show_pic(screen):
            if winner == GAME_PLAYER.PLAYER_ONE:
                screen.blit(self.play1_win, (self.conf.MAP_WIDTH + 25, self.conf.SCREEN_HEIGHT - 200))
            else:
                screen.blit(self.play2_win, (self.conf.MAP_WIDTH + 25, self.conf.SCREEN_HEIGHT - 200))

        show_pic(self.screen)
        pygame.mouse.set_visible(True)

    def change_mouse_show(self, empty_func):
        map_x, map_y = pygame.mouse.get_pos()
        x, y = map_pos_to_index(map_x, map_y, self.conf.REC_SIZE)
        if is_in_map(map_x, map_y, self.conf.MAP_WIDTH, self.conf.MAP_HEIGHT) and empty_func(y, x):
            pygame.mouse.set_visible(False)
            light_red = (21, 174, 103)
            pos, radius = (map_x, map_y), self.conf.CHESS_RADIUS
            pygame.draw.circle(self.screen, light_red, pos, radius)
        else:
            pygame.mouse.set_visible(True)

    def draw_background(self):
        color = (0, 0, 0)
        row, col = self.conf.CHESS_LEN, self.conf.CHESS_LEN
        for y in range(row):
            start_pos, end_pos = (
                (self.conf.REC_SIZE // 2, self.conf.REC_SIZE // 2 + self.conf.REC_SIZE * y),
                (self.conf.MAP_WIDTH - self.conf.REC_SIZE // 2, self.conf.REC_SIZE // 2 + self.conf.REC_SIZE * y))
            if y == row // 2:
                width = 2
            else:
                width = 1
            pygame.draw.line(self.screen, color, start_pos, end_pos, width)  # 画线
        for x in range(col):
            # draw a horizontal line
            start_pos, end_pos = (
                (self.conf.REC_SIZE // 2 + self.conf.REC_SIZE * x, self.conf.REC_SIZE // 2),
                (self.conf.REC_SIZE // 2 + self.conf.REC_SIZE * x, self.conf.MAP_HEIGHT - self.conf.REC_SIZE // 2)
            )
            if x == col // 2:
                width = 2
            else:
                width = 1
            pygame.draw.line(self.screen, color, start_pos, end_pos, width)
        rec_size = 8
        pos = [(3, 3), (11, 3), (3, 11), (11, 11), (7, 7)]
        for (x, y) in pos:
            pygame.draw.rect(
                self.screen,
                color,
                (
                    self.conf.REC_SIZE // 2 + x * self.conf.REC_SIZE - rec_size // 2,
                    self.conf.REC_SIZE // 2 + y * self.conf.REC_SIZE - rec_size // 2,
                    rec_size,
                    rec_size
                )
            )

    def draw_chess(self, steps, board):
        player_one = (10, 10, 10)  # 颜色rgb
        player_two = (255, 251, 240)
        player_color = {"one": player_one, "two": player_two}
        font = pygame.font.SysFont('simsunnsimsun', self.conf.REC_SIZE * 2 // 3)  # 棋盘表面
        for i in range(len(steps)):
            r, c = steps[i]  # 步长
            map_x, map_y, width, height = self.get_map_unit_rect(c, r)  # 棋盘上能下棋的位置
            pos, radius = (map_x + width // 2, map_y + height // 2), self.conf.CHESS_RADIUS  # 棋子半径
            current_player = board[r, c]
            if current_player == GAME_PLAYER.PLAYER_ONE:
                turn = "one"
                op_turn = "two"
            else:
                turn = "two"
                op_turn = "one"
            pygame.draw.circle(self.screen, player_color[turn], pos, radius)  # 画出圈
            msg_image = font.render(str(i), True, player_color[op_turn], player_color[turn])  # 双方的棋子
            msg_image_rect = msg_image.get_rect()  # 棋子反应
            msg_image_rect.center = pos  # 棋子中心
            self.screen.blit(msg_image, msg_image_rect)  # 屏幕刷新

    # 获取地图中可以下棋的单元
    def get_map_unit_rect(self, x, y):
        map_x = x * self.conf.REC_SIZE
        map_y = y * self.conf.REC_SIZE
        return map_x, map_y, self.conf.REC_SIZE, self.conf.REC_SIZE


class VisualGame(object):
    def __init__(self,
                 play1: Optional[Dict[str, Union[str, Callable]]] = None,
                 play2: Optional[Dict[str, Union[str, Callable]]] = None,
                 chess_size=15,
                 goal_chess_num=5
                 ):
        # 定义游戏棋盘的参数
        self.conf = VisualChessConfig(chess_size)
        human = {"method": self.human_play,
                 "name": "human"}
        if play1 is None:
            play1 = human
        if play2 is None:
            play2 = human
        # debug
        # play1 = human
        # play2 = human

        self.game = Game(chess_size, goal_chess_num, first_random_alpha=200)
        self.game.set_player(play1["method"], play2["method"])
        self.play_name = {GAME_PLAYER.PLAYER_ONE: play1["name"],
                          GAME_PLAYER.PLAYER_TWO: play2["name"]}
        self.displayer = Displayer(self.conf)
        self.train = True
        self.train_data = []

    def start(self):
        self.game.start()

    def end(self):
        self.game.end()

    def is_playing(self):
        return self.game.is_playing()

    def change_player_agent(self):
        self.game.change_player_agent()
        self.play_name = {GAME_PLAYER.PLAYER_ONE: self.play_name[GAME_PLAYER.PLAYER_TWO],
                          GAME_PLAYER.PLAYER_TWO: self.play_name[GAME_PLAYER.PLAYER_ONE]}
        logger.info("Player has been changed!")

    def human_play(self, chessboard):
        win_step = chessboard.search_current_player_certain_step()
        if len(win_step) > 0:
            print("win_step:", win_step, "self.current player:", chessboard.current_player)
        action = None
        while action is None:
            self.displayer.show_fps(self.game.chessboard.steps, self.game.chessboard.board,
                                    self.game.chessboard.is_empty, winner=self.game.winner)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    action = self.mouse_click(mouse_x, mouse_y)
                    status = self.check_buttons(mouse_x, mouse_y)
                    if status:
                        return None
        return action[::-1]

    def play(self, agent=None, wait=0, model_file=None):
        while True:
            # 手动控制开始对局
            self.displayer.show_fps(self.game.chessboard.steps, self.game.chessboard.board,
                                    self.game.chessboard.is_empty, winner=self.game.winner)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.check_buttons(mouse_x, mouse_y)
            if self.game.is_playing() and not self.game.is_over():
                self.train_data, winner = self.game.play(job_id=0, wait=wait, displayer=self.displayer)
            if self.game.is_over():
                self.click_button(self.displayer.buttons['surrend'])
                winner_name = self.play_name.get(self.game.winner)
                if len(self.train_data) > 0 and self.train and winner_name != "agent" and agent is not None:
                    loss = agent.train(self.train_data)
                    print("model_file:", model_file, "agent:", model_file)
                    agent.save_model(model_file)
                    logger.info(f"finished train, loss: {loss}")
                self.train_data = []

    # 处理鼠标输入
    def mouse_click(self, map_x, map_y):
        if self.game.is_playing() \
                and is_in_map(map_x, map_y, self.conf.MAP_WIDTH, self.conf.MAP_HEIGHT) \
                and not self.game.is_over():
            x, y = map_pos_to_index(map_x, map_y, self.conf.REC_SIZE)
            if self.game.is_empty(y, x):
                return x, y
            else:
                return None

    def click_button(self, button):
        if button.click(self):
            for tmp in self.displayer.buttons.values():
                if tmp != button:
                    tmp.unclick()

    def check_buttons(self, mouse_x, mouse_y):
        status = False
        for button in self.displayer.buttons.values():
            if button.rect.collidepoint(mouse_x, mouse_y):
                self.click_button(button)
                status = True
                break
        return status


# 地图上点的下标
def map_pos_to_index(map_x, map_y, rec_size):
    x = map_x // rec_size
    y = map_y // rec_size
    return x, y


# 判断是否在地图上
def is_in_map(map_x, map_y, map_width, map_height):
    if (map_x <= 0 or map_x >= map_width or
            map_y <= 0 or map_y >= map_height):
        return False
    return True


class Button(object):
    def __init__(self, screen, text, x, y, color, enable, conf):
        self.screen = screen
        self.width = conf.BUTTON_WIDTH
        self.height = conf.BUTTON_HEIGHT
        self.button_color = color
        self.text_color = (255, 255, 255)
        self.enable = enable
        self.font = pygame.font.SysFont('simsunnsimsun', conf.BUTTON_HEIGHT * 2 // 3)
        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.rect.topleft = (x, y)
        self.text = text
        self.init_msg()

    def init_msg(self):
        if self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
        else:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
        self.msg_image_rect = self.msg_image.get_rect()
        self.msg_image_rect.center = self.rect.center

    def draw(self):
        if self.enable:
            self.screen.fill(self.button_color[0], self.rect)
        else:
            self.screen.fill(self.button_color[1], self.rect)
        self.screen.blit(self.msg_image, self.msg_image_rect)


class StartButton(Button):
    def __init__(self, screen, text, x, y, conf):
        super().__init__(screen, text, x, y, [(107, 194, 53), (174, 221, 129)], True, conf)

    def click(self, game):
        if self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            game.start()
            return True
        return False

    def unclick(self):
        if not self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class ChangeUserButton(Button):
    def __init__(self, screen, text, x, y, conf):
        super().__init__(screen, text, x, y, [(107, 194, 53), (174, 221, 129)], True, conf)

    def click(self, game):
        if self.enable and not game.is_playing():
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            game.change_player_agent()
            return True
        return False

    def unclick(self):
        if not self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class GiveupButton(Button):
    def __init__(self, screen, text, x, y, conf):
        super().__init__(screen, text, x, y, [(254, 67, 101), (252, 157, 154)], False, conf)

    def click(self, game):
        if self.enable:
            game.end()
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):
        if not self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
            self.enable = True
