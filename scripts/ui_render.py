import cv2 as cv
import math
from time import localtime, strftime

user_color = (0, 255, 0)

def extend_angle(center, angle, radius):
    angle -= math.pi / 2
    return (int(center[0] + math.cos(angle) * radius), int(center[1] + math.sin(angle) * radius))

def render_ui(screen):
    screen_height, screen_width, _ = screen.shape
    render_clock(screen, 25, 170)

def render_clock(screen, x, y):
    current_time = localtime()
    date_time  = strftime("%a, %d %b %Y", current_time)
    clock_time = strftime("%H:%M:%S", current_time)

    circle_center = (x + 40, y - 100)
    circle_radius = 40

    cv.circle(screen,
        circle_center,
        circle_radius,
        user_color,
        2
    )

    time_seconds = current_time[5]
    seconds_length = circle_radius * 0.8
    seconds_angle  = (time_seconds / 60.) * math.pi * 2

    cv.line(screen,
        circle_center,
        extend_angle(circle_center, seconds_angle, seconds_length),
        user_color,
        1
    )

    time_minutes = current_time[4]
    minutes_length = circle_radius * 0.8
    minutes_angle  = (time_minutes / 60.) * math.pi * 2

    cv.line(screen,
        circle_center,
        extend_angle(circle_center, minutes_angle, minutes_length),
        user_color,
        2
    )

    time_hours = current_time[3]
    hours_length = circle_radius * 0.4
    hours_angle  = (time_hours / 12) * math.pi * 2

    cv.line(screen,
        circle_center,
        extend_angle(circle_center, hours_angle, hours_length),
        user_color,
        2
    )

    cv.putText(screen, clock_time,
        (x, y - 30), cv.FONT_HERSHEY_SIMPLEX,
        0.8, user_color, 2)

    cv.putText(screen, date_time,
        (x, y), cv.FONT_HERSHEY_SIMPLEX,
        0.8, user_color, 2)