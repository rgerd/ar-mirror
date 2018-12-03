import cv2 as cv
from time import gmtime, strftime

def render_ui(screen):
    screen_height, screen_width, _ = screen.shape
    render_clock(screen, 25, 200)

def render_clock(screen, x, y):
    date_time = strftime("%a, %d %b %Y", gmtime())
    clock_time = strftime("%H:%M:%S", gmtime())

    # cv.circle(screen, (x + 40, y - 100))

    cv.putText(screen, clock_time,
        (x, y), cv.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2)

    cv.putText(screen, date_time,
        (x, y + 20), cv.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2)