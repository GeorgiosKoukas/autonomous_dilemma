import carla
import pygame
import time
from pygame.locals import K_w, K_s, K_a, K_d

def handle_keyboard_input(control):
    keys = pygame.key.get_pressed()

    if keys[K_w]:
        control.throttle = 1.0
    else:
        control.throttle = 0

    if keys[K_s]:
        control.brake = 1.0
    else:
        control.brake = 0

    steer_increment = 0.1  # Adjust this value for smoother steering
    if keys[K_a]:
        control.steer = max(-1.0, control.steer - steer_increment)
    elif keys[K_d]:
        control.steer = min(1.0, control.steer + steer_increment)
    else:
        # Gradually reduce steering to 0 if no steering keys are pressed
        if control.steer < 0:
            control.steer = min(0, control.steer + steer_increment)
        elif control.steer > 0:
            control.steer = max(0, control.steer - steer_increment)

    return control

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Manual Control")
ticks = 0
control = carla.VehicleControl()

while ticks < 300:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    control = handle_keyboard_input(control)
    print(control)
    time.sleep(0.1)
    ticks += 1

pygame.quit()
