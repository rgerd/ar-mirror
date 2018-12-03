import cv2 as cv
import numpy as np

def render_ar(screen, PPI, user):
    screen_height, screen_width, _ = screen.shape
    perspective = list(user.predicted_position.value())
    color = (255, 255, 255)
    # fw, fh = user.measured_size
    render_menu(screen, PPI, perspective, color)
    render_face(screen, PPI, perspective, user.name, color)

def render_face(screen, PPI, perspective, name, color):
    # flip behind mirror
    point = perspective[:]
    point[2] = -point[2]

    p = mirror_point(perspective, point)
    p = frame_to_pixel(screen, PPI, p)
    org = (int(p[0]), int(p[1]))
    cv.circle(screen, org, int(32 - perspective[2]) * 2, color, 2)
    center = (org[0] - 40, org[1] - 40)
    cv.putText(screen, name, center, cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def render_menu(screen, PPI, perspective, color):
    obj = [[10, 10, -30],
           [10, 0, -30],
           [15, 0, -30],
           [15, 10, -30]]

    obj = [mirror_point(perspective, p) for p in obj]
    obj = [frame_to_pixel(screen, PPI, p) for p in obj]

    pts = np.array(obj, np.int32)
    pts = pts.reshape((-1,1,2))

    cv.polylines(screen,[pts],True,color)

def frame_to_pixel(screen, PPI, point):
    # axis offset (only along y)
    point = list(point)
    point[1] -= 5

    # scale to pixels
    point[0] *= PPI; point[1] *= PPI

    # convert to coordinates from top
    point[0] += screen.shape[1] / 2.
    point[1] = screen.shape[0] / 2. - point[1]

    return point

def mirror_point(perspective, location):
    if perspective[2] <= 0:
        print('invalid perspective: face is behind the mirror')
        return (0,0)
    if location[2] > 0:
        print('invalid location: positive z position for location... flipping sign')
        location[2] = -location[2]

    # calculate for z = 0
    (px, py, pz) = perspective
    (lx, ly, lz) = location
    x = px - ((px - lx) / float(pz - lz)) * pz
    y = py - ((py - ly) / float(pz - lz)) * pz

    return [x, y]
