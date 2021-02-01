from __future__ import print_function
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import os

# Directories / Image names

IMG_DIR = 'originals'
TMPL_DIR = 'templates2'
MATCH_RES = 1
MASK_DIR = 'masks2'
IMGN = 'StarMap.png'
TMPN = 'Small_area_rotated.png'


img = cv.imread(os.path.join(IMG_DIR, IMGN))
img2 = img.copy()
gray_img = cv.imread(os.path.join(IMG_DIR, IMGN), cv.IMREAD_GRAYSCALE)
template = cv.imread(os.path.join(IMG_DIR, TMPN), 0)
w, h = template.shape[::-1]


min_val = 1.23E+10
max_val = 1.23E+10
min_loc = None
max_loc = None
rotation = 0
res = None


def make_templates():
    """ Makes templates for rotational-deg=0...359 from base.jpg in TMPL_DIR.
    Saves rotated templates as tmpl{deg}.png in TMPL_DIR
    """
    try:
        base = cv.imread(os.path.join(IMG_DIR, TMPN))
    except IOError:
        print('Failed to make templates. Base template is not found')
        return
    for deg in range(360):
        tmpl = ndimage.rotate(base, deg)
        cv.imwrite(os.path.join(TMPL_DIR, 'tmpl%d.png' % deg), tmpl)
    return


def make_masks():
    """ Makes masks from tmpl{0...359}.png in TMPL_DIR.
    Saves masks as mask{0...359}.png in MASK_DIR
    """
    for deg in range(360):
        tmpl = cv.imread(os.path.join(TMPL_DIR, 'tmpl%d.png' % deg),
                         cv.IMREAD_GRAYSCALE)
        if tmpl is None:
            print('Failed to make mask {0}. tmpl{0}.png is not found.'.
                  format(deg))
        else:
            ret2, mask = cv.threshold(tmpl, 0, 255,
                                      cv.THRESH_BINARY+cv.THRESH_OTSU)
            cv.imwrite(os.path.join(MASK_DIR, 'mask%d.png' % deg), mask)
    return

# Function calls for making templates and masks (needs to be called only once)
# make_templates()
# make_masks()


for deg in range(0, 360, MATCH_RES):
    tmpl = cv.imread(os.path.join(TMPL_DIR, 'tmpl%d.png' %
                                  deg), cv.IMREAD_GRAYSCALE)
    mask = cv.imread(os.path.join(MASK_DIR, 'mask%d.png' % deg),
                     cv.IMREAD_GRAYSCALE)
    if tmpl is None or mask is None:
        print('Failed to match for tmpl %d.' % deg)
    else:

        temp_img = gray_img.copy()

        # Apply template Matching
        temp_res = cv.matchTemplate(
            temp_img, tmpl, cv.TM_SQDIFF_NORMED, mask=mask)
        temp_min_val, temp_max_val, temp_min_loc, temp_max_loc = cv.minMaxLoc(
            temp_res)

        if temp_min_val < min_val:
            min_val, max_val, min_loc, max_loc = temp_min_val, temp_max_val, temp_min_loc, temp_max_loc
            res = temp_res
            rotation = deg

top_left = min_loc

bottom_right = (top_left[0] + w, top_left[1] + h)
bottom_left = (top_left[0], top_left[1] + h)
top_right = (top_left[0] + w, top_left[1])
cv.rectangle(img, top_left, bottom_right, 255, 2)
plt.subplot(121), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle('Rotation: ' + str(rotation))

print(top_left)
print(top_right)
print(bottom_left)
print(bottom_right)
plt.show()
