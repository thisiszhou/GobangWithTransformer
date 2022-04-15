import pygame
# from pygame.locals import *
from chess_board.base_board import ChessBoard
from chess_board.cons import GAME_PLAYER
import numpy as np
from loguru import logger
from collections import defaultdict
import time

# 定义游戏棋盘的参数
REC_SIZE = 50  # 棋子移动空间的大小
CHESS_RADIUS = REC_SIZE // 2 - 2
CHESS_LEN = 15  # 棋盘长度
MAP_WIDTH = CHESS_LEN * REC_SIZE  # 地图宽度
MAP_HEIGHT = CHESS_LEN * REC_SIZE  # 地图高度
INFO_WIDTH = 200  # 信息宽度
BUTTON_WIDTH = 140  # 按钮宽度
BUTTON_HEIGHT = 50  # 按钮高度
SCREEN_WIDTH = MAP_WIDTH + INFO_WIDTH  # 屏幕宽度
SCREEN_HEIGHT = MAP_HEIGHT  # 屏幕高度
SEARCH_DEPTH = 5  # 搜索深度5
LIMITED_MOVE_NUM = 10  # 限制步数10


class Game(object):
    def __init__(self,
                 play1,
                 play2,
                 goal_chess_num,
                 bg_main_file,
                 bg_side_file,
                 play1_win_file,
                 play2_win_file,
                 show_board=True,
                 caption='Five Chess',
                 collect_train_data=False):
        """
        :param play1: method or str "human"
        :param play2: method or str "human"
        :param goal_chess_num:
        :param bg_main_file:
        :param bg_side_file:
        :param play1_win_file:
        :param play2_win_file:
        :param show_board:
        :param caption:
        """
        if not show_board:
            assert play1 != "human" and play2 != "human"
        if play1 == "human":
            play1 = self.human_play
        if play2 == "human":
            play2 = self.human_play
        self.player_method = {1: play1, -1: play2}
        self.current_player = None
        self.is_play = False
        self.chessboard = ChessBoard(CHESS_LEN, CHESS_LEN, goal_chess_num)
        self.action = None
        self.winner = None
        self.epoch = -1
        self.debug_show = True
        self.show_board = show_board
        if show_board:
            pygame.init()
            self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
            pygame.display.set_caption(caption)
            self.clock = pygame.time.Clock()
            self.buttons = {
                'start': StartButton(self.screen, 'start', MAP_WIDTH + 30, 15),
                'surrend': GiveupButton(self.screen, 'surrend', MAP_WIDTH + 30, BUTTON_HEIGHT + 45)
            }
            # background
            self.bg_main = pygame.image.load(bg_main_file)
            self.bg_side = pygame.image.load(bg_side_file)
            self.play1_win = pygame.image.load(play1_win_file)
            self.play2_win = pygame.image.load(play2_win_file)
        self.collect_train_data = collect_train_data
        self.board = []
        self.me = []
        self.oppo = []
        self.step = []
        self.label = []
        self.weight = []

    def start(self):
        self.is_play = True
        self.epoch += 1
        self.winner = None
        self.chessboard.reset()
        self.current_player = self.chessboard.current_player

        # collect data
        self.board = []
        self.me = []
        self.oppo = []
        self.step = []
        self.label = []

        if self.show_board:
            print("first:", self.player_method)

    def change_player_agent(self):
        self.player_method = {1: self.player_method[-1], -1: self.player_method[1]}

    def show_fps(self, fps):
        self.clock.tick(fps)
        self.screen.blit(self.bg_main, (0, 0))
        self.screen.blit(self.bg_side, (MAP_WIDTH, 0))
        for button in self.buttons.values():
            button.draw()
        if not self.is_over():
            self.changeMouseShow()
        if self.is_over():
            self.showWinner()
        self.draw_background()
        self.draw_chess()
        pygame.display.update()

    def human_play(self, chessboard):
        win_step = chessboard.search_current_player_certain_step()
        if len(win_step) > 0:
            print("win_step:", win_step, "self.current player:", chessboard.current_player)
        action = None
        while action is None:
            self.show_fps(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    action = self.mouseClick(mouse_x, mouse_y, chessboard)
                    self.check_buttons(mouse_x, mouse_y)
        return action[::-1]

    def play(self, train_agent=None, train_epoch=None, show_loss_step=None, save_model_step=None, save_model_file=None,
             wait=0.):
        """
        :return: board, me, oppo, step, label
        board: [step, last_steps, board_size, board_size]
        me: [step, half last_steps, board_size, board_size]
        oppo: [step, half last_steps, board_size, board_size]
        step: [step, 2]
        label: [step, 1]
        """
        epoch = 0
        total_loss = -1.
        winner_num = [0, 0]
        player_num = defaultdict(int)
        while True:
            # 手动控制开始对局
            if self.show_board:
                self.show_fps(600)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        self.check_buttons(mouse_x, mouse_y)
            # 自动控制开始对局
            elif not self.is_playing():
                self.start()

            if self.is_playing() and not self.is_over():
                if wait > 0:
                    time.sleep(wait)
                current_player = self.chessboard.current_player
                action = self.player_method[current_player](self.chessboard)
                self.move(*action)
                if self.collect_train_data:
                    self.update_train_data()
            if self.is_over():
                if self.show_board:
                    self.click_button(self.buttons['surrend'])
                if train_agent is not None and self.epoch == epoch:
                    player_num[str(self.player_method[self.winner])] += 1
                    if self.winner is GAME_PLAYER.PLAYER_ONE:
                        winner_num[0] += 1
                    else:
                        winner_num[1] += 1
                    if epoch < train_epoch:
                        # print("len:", len(self.get_train_data()[0]))
                        loss = train_agent.train(self.get_train_data())
                        self.change_player_agent()
                        epoch += 1
                        if total_loss < 0:
                            total_loss = loss
                        else:
                            total_loss = total_loss * 0.95 + loss * 0.05
                        if (epoch + 1) % show_loss_step == 0:
                            logger.info(f"Train epoch: {epoch}, loss: {total_loss}, winner times: {winner_num}, "
                                        f"player_num: {player_num}")
                        if (epoch + 1) % save_model_step == 0:
                            train_agent.save_model(save_model_file)
                    else:
                        break
                if train_agent is None and not self.show_board:
                    break
        return self.get_train_data()

    def update_train_data(self):
        board, me, oppo, step, winner = self.chessboard.get_last_train_window_output()
        # print("-"*20 + "train" + "-"*20)
        # print(board)
        # print(me)
        # print(oppo)
        # print("-"*20 + "train" + "-"*20)
        self.board.append(board)
        self.me.append(me)
        self.oppo.append(oppo)
        self.step.append(step)
        if winner is not None:
            length = len(self.board)
            win_length = int(length / 2 + length % 2)
            lose_length = int(length / 2)
            self.label = np.zeros((length, 1), dtype=np.float)
            self.weight = np.zeros((length, 1), dtype=np.float)
            self.label[1 - length % 2::2, 0] = 1
            self.weight[1 - length % 2::2, 0] = np.array([0.5 + x / (win_length * 2) for x in range(1, win_length + 1)])
            self.weight[length % 2::2, 0] = 1 / lose_length

    def get_train_data(self):
        return self.board, self.me, self.oppo, self.step, self.label, self.weight

    def changeMouseShow(self):
        map_x, map_y = pygame.mouse.get_pos()
        x, y = self.MapPosToIndex(map_x, map_y)
        if self.isInMap(map_x, map_y) and self.chessboard.is_empty(y, x):
            pygame.mouse.set_visible(False)
            light_red = (21, 174, 103)
            pos, radius = (map_x, map_y), CHESS_RADIUS
            pygame.draw.circle(self.screen, light_red, pos, radius)
        else:
            pygame.mouse.set_visible(True)

    def move(self, r, c):
        status, winner = self.chessboard.move(r, c)
        if status == -1:
            raise ValueError("status is -1")
        if status == 0:
            self.player = self.chessboard.current_player
            self.debug_show = True
        elif status == 2:
            self.winner = GAME_PLAYER.PLAYER_TWO
            self.is_play = False
        else:
            self.winner = winner
            self.is_play = False

    # 处理鼠标输入
    def mouseClick(self, map_x, map_y, chessboard: ChessBoard):
        if self.is_play and self.isInMap(map_x, map_y) and not self.is_over():
            x, y = self.MapPosToIndex(map_x, map_y)
            if chessboard.is_empty(y, x):
                return x, y
            else:
                return None

    def is_over(self):
        return self.winner is not None

    def is_playing(self):
        return self.is_play

    def showWinner(self):
        def showPIC(screen):
            if self.winner == GAME_PLAYER.PLAYER_ONE:
                screen.blit(self.play1_win, (MAP_WIDTH + 25, SCREEN_HEIGHT - 200))
            else:
                screen.blit(self.play2_win, (MAP_WIDTH + 25, SCREEN_HEIGHT - 200))

        showPIC(self.screen)
        pygame.mouse.set_visible(True)

    def click_button(self, button):
        if button.click(self):
            for tmp in self.buttons.values():
                if tmp != button:
                    tmp.unclick()

    def check_buttons(self, mouse_x, mouse_y):
        for button in self.buttons.values():
            if button.rect.collidepoint(mouse_x, mouse_y):
                self.click_button(button)
                break

    def draw_background(self):
        color = (0, 0, 0)
        row, col = self.chessboard.shape()
        for y in range(row):
            # draw a horizontal line
            start_pos, end_pos = (
                (REC_SIZE // 2, REC_SIZE // 2 + REC_SIZE * y),
                (MAP_WIDTH - REC_SIZE // 2, REC_SIZE // 2 + REC_SIZE * y))  # 棋盘位置
            if y == row // 2:
                width = 2
            else:
                width = 1
            pygame.draw.line(self.screen, color, start_pos, end_pos, width)  # 画线
        for x in range(col):
            # draw a horizontal line
            start_pos, end_pos = (
                (REC_SIZE // 2 + REC_SIZE * x, REC_SIZE // 2),
                (REC_SIZE // 2 + REC_SIZE * x, MAP_HEIGHT - REC_SIZE // 2)
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
                    REC_SIZE // 2 + x * REC_SIZE - rec_size // 2,
                    REC_SIZE // 2 + y * REC_SIZE - rec_size // 2,
                    rec_size,
                    rec_size
                )
            )

    def draw_chess(self):
        player_one = (10, 10, 10)  # 颜色rgb
        player_two = (255, 251, 240)
        player_color = {"one": player_one, "two": player_two}
        font = pygame.font.SysFont('simsunnsimsun', REC_SIZE * 2 // 3)  # 棋盘表面
        steps = self.chessboard.get_steps()
        for i in range(len(steps)):
            r, c = steps[i]  # 步长
            map_x, map_y, width, height = self.getMapUnitRect(c, r)  # 棋盘上能下棋的位置
            pos, radius = (map_x + width // 2, map_y + height // 2), CHESS_RADIUS  # 棋子半径
            current_player = self.chessboard.get_chess_value(r, c)
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
        self.debug_show = False
        # if len(steps) > 0:
        #     last_pos = steps[-1]
        #     map_x, map_y, width, height = self.getMapUnitRect(last_pos[1], last_pos[0])
        #     purple_color = (0, 0, 255)
        #     point_list = [(map_x, map_y), (map_x + width, map_y),
        #                   (map_x + width, map_y + height), (map_x, map_y + height)]
        #     pygame.draw.lines(self.screen, purple_color, True, point_list, 1)

    # 获取地图中可以下棋的单元
    def getMapUnitRect(self, x, y):
        map_x = x * REC_SIZE
        map_y = y * REC_SIZE
        return map_x, map_y, REC_SIZE, REC_SIZE

    # 地图上点的下标
    def MapPosToIndex(self, map_x, map_y):
        x = map_x // REC_SIZE
        y = map_y // REC_SIZE
        return x, y

    # 判断是否在地图上
    def isInMap(self, map_x, map_y):
        if (map_x <= 0 or map_x >= MAP_WIDTH or
                map_y <= 0 or map_y >= MAP_HEIGHT):
            return False
        return True


class Button(object):
    def __init__(self, screen, text, x, y, color, enable):
        self.screen = screen
        self.width = BUTTON_WIDTH
        self.height = BUTTON_HEIGHT
        self.button_color = color
        self.text_color = (255, 255, 255)
        self.enable = enable
        self.font = pygame.font.SysFont('simsunnsimsun', BUTTON_HEIGHT * 2 // 3)
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
    def __init__(self, screen, text, x, y):
        super().__init__(screen, text, x, y, [(107, 194, 53), (174, 221, 129)], True)

    def click(self, game):
        if self.enable:
            game.start()
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):
        if not self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
            self.enable = True


class GiveupButton(Button):
    def __init__(self, screen, text, x, y):
        super().__init__(screen, text, x, y, [(254, 67, 101), (252, 157, 154)], False)

    def click(self, game):
        if self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[1])
            self.enable = False
            return True
        return False

    def unclick(self):
        if not self.enable:
            self.msg_image = self.font.render(self.text, True, self.text_color, self.button_color[0])
            self.enable = True
