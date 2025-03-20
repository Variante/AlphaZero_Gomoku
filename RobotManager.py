#!/usr/bin/env python3

import os
import sys
import time
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from uarm.wrapper import SwiftAPI
import numpy as np
import cv2
import random
import threading
import scipy.stats
import apriltag
import traceback

"""
Position Coordinates:
Reset: (200, 0, 150)
     ^ x (150, 310)
     |
y<---|--- (-120, 120)
z: height (30, 100?)
Grap height: z=25
"""


class RobotManager:
    def __init__(self, b_width, b_height, obs_size=100):
        self.swift = None

        x_limit = (130, 310)
        y_limit = (-100, 100)
        z_limit = (5, 80)

        self.limits = [x_limit, y_limit, z_limit]
        self.action_scale = np.array([10, 10, 10])
        self.speed = 100000
        self.pos = None

        self.cam_top = None
        self.cam_thread = None

        # array for calibration
        calib_x_num = 5
        calib_y_num = 5
        calib_z_num = 3
        self.calib_3d = np.array(
            [[[(i // calib_x_num) * (self.limits[0][1] - self.limits[0][0]) / (calib_y_num - 1) + self.limits[0][0],
               (i % calib_x_num - calib_x_num // 2) * (self.limits[1][1] - self.limits[1][0]) / (calib_x_num - 1),
               25 + j * 20] for j in range(calib_z_num)] for i in range(calib_x_num * calib_y_num)]
        )
        self.calib_3d = self.calib_3d.reshape(-1, 3)
        self.calib_2d = np.array([[0, 0] for _ in range(self.calib_3d.shape[0])])
        self.board_2d = []
        self.calib_p = None
        self.frame_top = None
        self.corners = None
        self.pickup_p = (115, -171) # location to pickup the checker
        self.pickup_c = 0 # counter for num of checkers we picked it up
        """
        options = apriltag.Detectoroptions(families='tag36h11',
                                 border=1,
                                 nthreads=4,
                                 quad_decimate=1.0,
                                 quad_blur=0.0,
                                 refine_edges=True,
                                 refine_decode=False,
                                 refine_pose=False,
                                 debug=False,
                                 quad_contours=True)
        """
        self.tagdetector = apriltag.Detector()
        
        self.b_width = b_width  # checkerboard size
        self.b_height = b_height

    def get_pos(self):
        # res = self.swift.get_polar()
        res = self.swift.get_position()
        assert isinstance(res, list)
        self.pos = np.array(res)
        return self.pos

    def __enter__(self):
        swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
        swift.waiting_ready(timeout=3)
        device_info = swift.get_device_info()
        print(device_info)
        firmware_version = device_info['firmware_version']
        if firmware_version and not firmware_version.startswith(('0.', '1.', '2.', '3.')):
            swift.set_speed_factor(0.0005)

        swift.set_mode(0)
        self.swift = swift
        self.cap_top = cv2.VideoCapture(0) # for macos it may open your build-in camera, just stop and launch the code again
        if not self.cap_top.isOpened():
            raise IOError("Cannot open webcam")
        # self.is_running = True
        # self.cam_thread = threading.Thread(target=self.cam_thread_func)
        # self.cam_thread.start()
        self.reset()
        return self

    def __exit__(self, *arg, **kwargs):
        print('Disconnected')
        self.swift.disconnect()
        self.is_running = False
        if self.cam_thread:
            self.cam_thread.join()
        self.cap_top.release()

    def _calib(self):
        # print(self.calib_3d, self.calib_3d.dtype)
        # print(self.calib_2d, self.calib_2d.dtype)
        # guess camera matrix first
        camera_matrix = np.zeros((3, 3), 'float32')
        camera_matrix[0, 0] = (self.frame_top.shape[1] + self.frame_top.shape[2]) / 2
        camera_matrix[1, 1] = (self.frame_top.shape[1] + self.frame_top.shape[2]) / 2
        camera_matrix[2, 2] = 1.0

        camera_matrix[0, 2] = self.frame_top.shape[2] / 2
        camera_matrix[1, 2] = self.frame_top.shape[1] / 2

        objpoints = [self.calib_3d.astype(np.float32)]
        imgpoints = [self.calib_2d[:, :2].astype(np.float32)]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                           self.frame_top.shape[1:][::-1], camera_matrix, None,
                                                           flags=cv2.CALIB_USE_INTRINSIC_GUESS
                                                           )
        assert ret

        mean_error = 0
        for i in range(1):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            # print(imgpoints2.shape, imgpoints[i].shape)
            imgpoints2 = imgpoints2.reshape(-1, 2)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("total error: {}".format(mean_error / len(objpoints)))
        self.calib_p = (mtx, dist, rvecs[0], tvecs[0])

    def query_position(self, img_pts, z=-60):
        mtx, dist, rvec, tvec = self.calib_p

        rmat, _ = cv2.Rodrigues(rvec)

        # print(rmat.shape, tvec.shape)

        img_pts = img_pts.reshape(-1, 2)
        undistorted_pts = cv2.undistortPoints(img_pts, mtx, dist, P=mtx)
        # print(undistorted_pts, img_pts)

        p = np.dot(mtx, np.hstack([rmat, tvec.reshape(3, -1)]))
        u = undistorted_pts[0, 0, 0]
        v = undistorted_pts[0, 0, 1]

        pz34 = p[:, 2] * z + p[:, 3]
        b = np.array([pz34[2] * u - pz34[0],
                      pz34[2] * v - pz34[1]])
        a = np.array([[p[0, 0] - u * p[2, 0], p[0, 1] - u * p[2, 1]],
                      [p[1, 0] - v * p[2, 0], p[1, 1] - v * p[2, 1]]]
                     )

        return np.linalg.solve(a, b)

    def load_calib(self, file='calib.npy'):
        try:
            calib = np.load(file)
        except:
            self.calibrate()
            return
        # print(calib.shape)
        assert calib.shape[0] > 0 and calib.shape[1] > 4
        self.calib_3d = calib[:, :3]
        self.calib_2d = calib[:, 3:]
        self._calib()
        # print("Load calib:")
        # print(self.calib_3d)
        # print(self.calib_2d)
        # print(self.calib_p)

    def detect_tag(self, img):
        gray = cv2.cvtColor(self.frame_top, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        result = self.tagdetector.detect(cl)
        return result

    def capture_img(self):
        while True:
            for _ in range(3):
                ret, frame_top = self.cap_top.read()
            if ret:
                break
            print('Image read error, retry in 1s')
            time.sleep(1)
        # print(frame_top.shape)
        self.frame_top = frame_top.copy()
        
        def to_cv(t):
            return int(t[0]), int(t[1])

        for i in self.calib_2d:
            if np.sum(i) > 0:
                frame_top = cv2.circle(frame_top, to_cv(i), 3, (255, 0, 0), 2)
        for i in self.board_2d:
            frame_top = cv2.circle(frame_top, to_cv(i), 5, (0, 255, 0), 3)
            
        cv2.imshow('Input', frame_top)
        cv2.waitKey(10)

    def warp_board(self, tags):
        cl = [None] * 4
        for i in tags:
            if i.tag_id != 0:
                cl[i.tag_id - 1] = i.center
        for i in cl:
            if i is None:
                print("Can not detect 4 corners")
                print(self.corners)
                time.sleep(1)
                if self.corners is None:
                    return None
                else:
                    cl = self.corners
                    break
        else:
            self.corners = cl

        def order_points(pts):
            # initialzie a list of coordinates that will be ordered
            # such that the first entry in the list is the top-left,
            # the second entry is the top-right, the third is the
            # bottom-right, and the fourth is the bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            # the top-left point will have the smallest sum, whereas
            # the bottom-right point will have the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            # now, compute the difference between the points, the
            # top-right point will have the smallest difference,
            # whereas the bottom-left will have the largest difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            # return the ordered coordinates
            return rect

        def four_point_transform(image, pts):
            # obtain a consistent order of the points and unpack them
            # individually
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            """
            # compute the width of the new image, which will be the
            # maximum distance between bottom-right and bottom-left
            # x-coordiates or the top-right and top-left x-coordinates
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            # compute the height of the new image, which will be the
            # maximum distance between the top-right and bottom-right
            # y-coordinates or the top-left and bottom-left y-coordinates
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            # now that we have the dimensions of the new image, construct
            # the set of destination points to obtain a "birds eye view",
            # (i.e. top-down view) of the image, again specifying points
            # in the top-left, top-right, bottom-right, and bottom-left
            # order
            """
            maxWidth = 400
            maxHeight = 300
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            # compute the perspective transform matrix and then apply it
            M = cv2.getPerspectiveTransform(rect, dst)
            invM = cv2.getPerspectiveTransform(dst, rect)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            # return the warped image
            return warped, invM

        return four_point_transform(self.frame_top, np.array(cl))

    def img_to_board(self, warp):
        img, invM = warp

        def parse_color(m, px, py):
            b = 3
            hsv = [np.mean(m[py-b:py+b, px-b:px+b, t]) for t in range(3)]
            # print(hsv)
            if hsv[1] > 100:
                if hsv[0] < 40:
                    return 2
                elif hsv[0] > 70:
                    return 1
            return 0

        w = self.b_width
        h = self.b_height
        # availables = []
        states = {0: [], 1: [], 2:[]}
        wleft = 60
        ww = 53
        htop = 12
        hh = 55

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # pts = []
        self.board_2d = []
        for j in range(h):
            for i in range(w):
                px = i * ww + wleft
                py = j * hh + htop
                stone = parse_color(hsv, px, py)
                idx = j * w + i
                if stone == 0:
                    color = (128, 128, 128)
                    # availables.append(idx)
                else:
                    color = (255, 0, 0) if stone == 1 else (0, 200, 255)
                states[stone].append(idx)
                # pts.append([px, py])
                projpt = np.dot(invM, np.array([px, py, 1]))
                projpt = projpt / projpt[2]
                # print(projpt, projpt.shape)
                self.board_2d.append(projpt[:2])
                img = cv2.circle(img, (px, py), 3, color, 2)

        # self.board_2d = cv2.perspectiveTransform(np.array(pts), invM)
        # print(self.board_2d)
        return img, states

    def parse_board(self):
        pos = self.get_pos()
        if (pos[0] - self.pickup_p[0]) ** 2 + (pos[1] - self.pickup_p[1]) ** 2 > 10:
            self.swift.set_position(x=self.pickup_p[0], y=self.pickup_p[1], z=70, wait=True, speed=1000000)
            time.sleep(2)
        warp = None
        while warp is None:
            self.capture_img()
            tags = self.detect_tag(self.frame_top)
            warp = self.warp_board(tags)
        parse, states = self.img_to_board(warp)
        if warp is not None:
            cv2.imshow('Warp', parse)
            cv2.waitKey(10)
        print('-' * 5)
        return states

    def query_position(self, img_pts, z=-70):
        mtx, dist, rvec, tvec = self.calib_p

        rmat, _ = cv2.Rodrigues(rvec)

        # print(rmat.shape, tvec.shape)

        img_pts = img_pts.reshape(-1, 2)
        undistorted_pts = cv2.undistortPoints(img_pts, mtx, dist, P=mtx)
        # print(undistorted_pts, img_pts)

        p = np.dot(mtx, np.hstack([rmat, tvec.reshape(3, -1)]))
        u = undistorted_pts[0, 0, 0]
        v = undistorted_pts[0, 0, 1]

        pz34 = p[:, 2] * z + p[:, 3]
        b = np.array([pz34[2] * u - pz34[0],
                      pz34[2] * v - pz34[1]])
        a = np.array([[p[0, 0] - u * p[2, 0], p[0, 1] - u * p[2, 1]],
                      [p[1, 0] - v * p[2, 0], p[1, 1] - v * p[2, 1]]]
                     )

        return np.linalg.solve(a, b)

    def calibrate(self):
        print("Calibration start")
        for j, i in enumerate(self.calib_3d):
            # print("Move to ", i)
            self.swift.set_position(x=i[0], y=i[1], z=i[2], wait=True, speed=50000)
            time.sleep(1)
            while True:
                # res = locate_cube_top(self.frame_top, dot_h_func_top)
                self.capture_img()
                results = self.detect_tag(self.frame_top)
                # self.check_corners(results)
                for i in results:
                    if i.tag_id == 0:
                        res = i.center
                        break
                else:
                    res = None
                if res is not None:
                    # print("Visual position: ", res)
                    self.calib_2d[j, 0] = res[0]
                    self.calib_2d[j, 1] = res[1]
                    # time.sleep(2)
                    break
                else:
                    print("Can not detect tag")
                time.sleep(1)
            """
            while True:
                res = locate_cube(self.frame, dot_h_func)
                self.dot_pts = res
                if res is not None:
                    print("Visual position: ", res)
                    self.calib_2d[j, 2] = res[0]
                    self.calib_2d[j, 3] = res[1]
                    # time.sleep(2)
                    break
                
            """
        calib = np.hstack([self.calib_3d, self.calib_2d])
        np.savetxt("calib.csv", calib, delimiter=',', header="X,Y,Z,PX,PY")
        np.save("calib.npy", calib)
        self._calib()
        print("Calibration done")
        self.swift.reset()

    def cam_thread_func(self):
        # do not play with UI in the spawn thread!
        while self.is_running:
            while True: 
                ret, frame_top = self.cap_top.read()
                if ret:
                    break
                time.sleep(0.1)
            # print(frame_top.shape)
            self.frame_top = frame_top.copy()
            continue

            def to_cv(t):
                return int(t[0]), int(t[1])

            for i in self.calib_2d:
                if np.sum(i) > 0:
                    frame_top = cv2.circle(frame_top, to_cv(i), 5, (255, 0, 0), 3)

            for i in self.board_2d:
                # print(i)
                frame_top = cv2.circle(frame_top, to_cv(i), 5, (0, 255, 0), 3)
            """
            if self.cube_pts is not None:
                frame = cv2.circle(frame, to_cv(self.cube_pts), 3, (0, 0, 255), 2)
            if self.dot_pts is not None:
                frame = cv2.circle(frame, to_cv(self.dot_pts), 3, (0, 255, 0), 2)
            for i in self.calib_2d:
                if np.sum(i) > 0:
                    frame = cv2.circle(frame, to_cv(i[2:]), 2, (255, 0, 0), 1)
            """
            # print(frame_top.shape)
            # cv2.imshow('Input', frame_top)
            # prepare observation
            # h = frame.shape[0]
            # w = frame.shape[1]
            # c = (w - h) // 2

            # cv2.imshow('Obs', np.concatenate([self.obs[:,:,:3], self.obs[:,:,3:]], axis=1))
            # cv2.imshow('Obs2', self.obs[:, :, 3:])
            # cv2.waitKey(10)
        # cv2.destroyAllWindows()

    def pickup(self):
        self.swift.set_position(x=int(self.pickup_p[0]), y=int(self.pickup_p[1]), z=max(0, 32 - self.pickup_c),
                                wait=True,
                                speed=1000000)
        time.sleep(1)
        self.swift.set_pump(True)
        time.sleep(1)
        self.swift.set_position(x=int(self.pickup_p[0]), y=int(self.pickup_p[1]), z=60, wait=True, speed=1000000)
        self.pickup_c += 1

    def drop_at(self, move):
        try:
            self.pickup()
            # print(move, self.corners)
            img_pos = self.board_2d[move]
            pos = self.query_position(img_pos)
            # print(img_pos, pos)
            self.swift.set_position(x=int(pos[0]), y=int(pos[1]), z=2, wait=True, speed=500000)
            time.sleep(3)
            self.swift.set_pump(False)
            time.sleep(1)
            self.swift.set_position(x=int(pos[0]), y=int(pos[1]), z=30, wait=True, speed=1000000)
            self.swift.set_position(x=self.pickup_p[0], y=self.pickup_p[1], z=70, wait=True, speed=1000000)
        except IndexError:
            self.swift.set_pump(False)

    def flip(self):
        # self.win()
        self.parse_board()
        pos = self.query_position((self.corners[1] + self.corners[0]) / 2)
        self.swift.set_position(x=int(pos[0] + 5), y=int(pos[1]), z=10, wait=True, speed=1000000)
        self.swift.set_position(x=int(pos[0] + 10), y=int(pos[1]), z=-5, wait=True, speed=1000000)
        time.sleep(1)
        self.swift.set_pump(True)
        time.sleep(1)
        self.swift.set_position(x=150, y=0, z=150, wait=True, speed=100000000)
        time.sleep(2)
        self.swift.set_pump(False)
        self.win()

    def win(self):
        for i in range(10):
            self.swift.set_position(x=150, y=20 * (1 if i % 2 else -1), z=150, wait=True, speed=100000000)

    def reset(self):
        self.swift.set_position(x=self.pickup_p[0], y=self.pickup_p[1], z=70, wait=True, speed=1000000)
        self.pickup_c = 0


if __name__ == '__main__':
    with RobotManager() as env:
        # env.calibrate()
        # env.capture_img()
        # env.load_calib()
        for i in range(18):
            env.parse_board()
            env.drop_at(i)
