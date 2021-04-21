import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def write(file_name, d, labels):
    """
    Write out a csv. Each element in d is a line in the file.
    :param labels:      The csv labels in list of string
    :param file_name:   The filename to write out
    :param d:           The data to write out
    """
    with open(file_name, "w") as file:
        file.write(",".join(labels) + "\n")
        for dd in d:
            s_list = [str(elem) for elem in dd]
            s_list = ",".join(s_list) + "\n"
            file.write(s_list)


def sample(im, num_points=50, r=20):
    """
    Sample uniformly at random num_points samples with radius r.
    class probability is (p1, p2, p3) = (building, road, nothing)

    :param im:          The image
    :param num_points:  The number of points to sample
    :param r:           The radius of each point
    :return:            list[point_x, point_y, r, p1, p2, p3]
    """
    w = img.shape[1]
    h = img.shape[0]
    mask = np.zeros([2*r, 2*r], dtype=np.uint8)
    mask = cv.circle(mask, (int(mask.shape[1]/2), int(mask.shape[0]/2)+1), r+1, color=(255, 255, 255), thickness=-1)
    mask[mask == 0] = 1

    points = []
    for i in range(num_points):
        pt = np.array([np.random.randint(r, w-r), np.random.randint(r, h-r)])
        too_close = False
        for p in points:
            if np.linalg.norm(p[0:2]-pt) <= 2*r:
                too_close = True
        if too_close:
            continue

        t = im[pt[1]-r:pt[1]+r, pt[0]-r:pt[0]+r]
        masked_img = cv.bitwise_or(t, t, mask=mask)
        classes = np.zeros(3)
        for x in range(2*r):
            for y in range(2*r):
                rr = np.linalg.norm(np.array([r, r], dtype=np.uint8)-np.array([x,y]))
                if rr <= r:
                    c = masked_img[y, x]  # opencv uses BGR convention!
                    if c[0] >= 200:  # Blue
                        classes[1] += 1
                    elif c[2] >= 200:  # Red
                        classes[0] += 1
                    elif c[0] == 0 and c[1] == 0 and c[2] == 0:
                        classes[2] += 1

        class_prob = np.around(classes/sum(classes), 2)
        points.append([pt[0], pt[1], r, class_prob[0], class_prob[1], class_prob[2]])

        # Print out the image.
        im = cv.circle(img, (pt[0], pt[1]), r, color=(0, 255, 0), thickness=1)
        font = cv.FONT_HERSHEY_SIMPLEX
        org = (pt[0]-2*r, pt[1])
        font_scale = 0.5
        color = (255,255,255)
        thickness = 1
        cv.putText(im, str(list(class_prob)), org, font, font_scale, color, thickness, cv.LINE_AA)
    return points


if __name__ == "__main__":

    # load image
    img = cv.imread('images/segments.png')
    # scale image
    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv.resize(img, dim)

    # TODO sample uniformly over classes, not (x,y) points

    data = sample(img)
    write("test.csv", data, ["x", "y", "r", "p(building)", "p(road)", "p(none)"])

    cv.imshow("", img)
    cv.waitKey(delay=100)
    plt.plot(0, 0, label="[p(building), p(road), p(none)]", color='green')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.legend()
    plt.show()