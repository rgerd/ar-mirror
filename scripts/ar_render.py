import cv2 as cv

def render_ar(screen, PPI, user):
    screen_height, screen_width, _ = screen.shape
    perspective = list(user.predicted_position.value())
    # fw, fh = user.measured_size

    obj = list(perspective)
    obj[2] = -obj[2]
    render_menu(screen, PPI, perspective, obj)

def render_menu(screen, PPI, perspective, point):
    loc = mirror_point(perspective, point)
    loc = frame_to_pixel(screen, PPI, loc)
    cv.circle(screen,
        (int(loc[0]), int(loc[1])),
        int(perspective[2]/ 4.),
        (255, 255, 255),
        2
    )

def frame_to_pixel(screen, PPI, point):
    PPI += 20 # ~M A G I C~

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
