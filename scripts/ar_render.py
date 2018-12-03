import cv2 as cv

def render_ar(screen):
    screen_height, screen_width, _ = screen.shape
    render_menu(screen, (0, 0, -.3), (0, 0, 3))

def render_menu(screen, perspective, point):
    cv.circle(screen,
        (int(screen.shape[0] / 2.), int(screen.shape[1] / 2.)),
        10,
        (255, 255, 255),
        2
    )


