from chess_board.visual_game import VisualGame
import pygame
from loguru import logger


game = VisualGame(1, 1, 4, "bgmain.jpg", "bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg")
while True:
    game.play()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            game.mouseClick(mouse_x, mouse_y)
            game.check_buttons(mouse_x, mouse_y)