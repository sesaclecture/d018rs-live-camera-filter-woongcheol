import cv2
import numpy as np


class Filters:
    Kernels = {
        "Original": np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]], dtype=np.float32),
        "Blur": np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.float32) / 9,
        "Gaussian blur": np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]], dtype=np.float32) / 16,
        "Sharpen": np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], dtype=np.float32),
        "Sobel (X)": np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32),
        "Sobel (Y)": np.array([[-1, -2, -1],
                             [0,  0,  0],
                             [1,  2,  1]], dtype=np.float32),
        "Edge detection": np.array([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]], dtype=np.float32),
        "Emboss": np.array([[-2, -1, 0],
                            [-1,  1, 1],
                            [ 0,  1, 2]], dtype=np.float32)
    }       

    def __init__(self, kernels=Kernels):
        self.kernels = kernels
        self.current_filter_number = 0

    def apply_filter(self, frame, filter_name) -> np.array:
        kernel = self.kernels[filter_name]
        return cv2.filter2D(frame, -1, kernel)

    def get_current_filter_name(self) -> str:
        keys = list(self.kernels.keys())
        return keys[self.current_filter_number]

    def switch_next_filter(self):
        if self.current_filter_number < 7:
            self.current_filter_number += 1
        else:
            self.current_filter_number = 0

    def switch_previous_filter(self):
        if self.current_filter_number > 0:
            self.current_filter_number -= 1
        else:
            self.current_filter_number = 7