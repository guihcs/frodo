import pygame
from board3 import Board3
from controller3 import ActionController

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0
t = 0
board = Board3()
controller = ActionController(board)

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    board_repr = board.board_repr()

    sw = 80
    lp = screen.get_width() / 2 - len(board_repr[0]) * sw / 2
    tp = screen.get_height() / 2 - len(board_repr) * sw / 2 - 100

    for i in range(len(board_repr)):
        for j in range(len(board_repr[i])):
            rect = pygame.Rect(j * sw + lp, i * sw + tp, sw, sw)
            if board_repr[i][j] == -1:
                pygame.draw.rect(screen, "blue", rect)
            elif board_repr[i][j] == 0:
                pygame.draw.rect(screen, "yellow", rect)
            elif board_repr[i][j] == 1:
                pygame.draw.rect(screen, "green", rect)
            elif board_repr[i][j] == 2:
                pygame.draw.rect(screen, "red", rect)
            elif board_repr[i][j] == 3:
                pygame.draw.rect(screen, "white", rect)
            elif board_repr[i][j] == 4:
                pygame.draw.rect(screen, "cyan", rect)




    keys = pygame.key.get_pressed()
    pos = pygame.mouse.get_pos()
    pressed = pygame.mouse.get_pressed()

    print(pos, pressed)

    if keys[pygame.K_w]:
        controller.execute_action_from_name('Move -1 0')

    if keys[pygame.K_s]:
        controller.execute_action_from_name('Move 1 0')

    if keys[pygame.K_a]:
        controller.execute_action_from_name('Move 0 -1')

    if keys[pygame.K_d]:
        controller.execute_action_from_name('Move 0 1')



    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()