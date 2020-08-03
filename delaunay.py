# !/usr/bin/python
import dlib
import cv2
import numpy as np
import random


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    affine_points = []
    for i, t in enumerate(triangleList):

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if not i:
            cv2.circle(img, pt1, 3, (150,150, 150))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)



# Draw voronoi diagram
def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)


if __name__ == '__main__':

    # Define window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"

    # Turn on animation while drawing triangles
    animate = True

    detector = dlib.get_frontal_face_detector()
    # Predictor pre-trained model can be downloaded from:
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

    try:
        predictor = dlib.shape_predictor('../shape_predictor_81_face_landmarks.dat')
    except RuntimeError as e:
        print('ERROR:  \'shape_predictor_68_face_landmarks.dat\' was not found in current directory.   ' \
              'Download it from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2')


    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)

    # Read in the image.
    img = cv2.imread("../0.jpg")

    # Keep a copy around
    img_orig = img.copy()

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);

    # Create an array of points.
    points = []

    faces = detector(img_orig, 0)
    if len(faces) >= 1:
        face_points = predictor(img_orig, faces[0])
        # print(face_points.parts())
        for part in face_points.parts():
            points.append((int(part.x), int(part.y)))
    points = points[27:36]
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

        # Show animation
        if animate:
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay(img_copy, subdiv, (255, 255, 255));
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)

    # Draw delaunay triangles
    draw_delaunay(img, subdiv, (255, 255, 255));

    # Draw points
    for p in points:
        draw_point(img, p, (0, 0, 255))
        



    # Show results
    cv2.imshow(win_delaunay, img)
    # cv2.imshow(win_voronoi, img_voronoi)
    cv2.waitKey(0)
