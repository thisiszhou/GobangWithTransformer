from chess_board.game import Game
import pygame
from loguru import logger


game = Game(None, None, 5, "bg_picture/bgmain.jpg", "bg_picture/bgbian.jpg", "bg_picture/manwin.jpg", "bg_picture/comwin.jpg")
while True:
    print("in play")
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