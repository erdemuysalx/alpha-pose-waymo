from builtins import exit
from email.mime import image
from itertools import count
from threading import Thread
from queue import Queue
import json

import cv2
import numpy as np

import torch
import torch.multiprocessing as mp

from alphapose.utils.presets import SimpleTransform

import os
import io
import logging
import tensorflow as tf
import PIL.Image
from PIL import Image
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import keypoint_data

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# logger = logging.getLogger("WaymoDetectionLoader")


def ddict():
    """
    Dummy function to create nested dictionaries.
    Replacement for lambda which causes pickling
    errors used by multiprocessing module.
    """
    return defaultdict(set)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


class WaymoDetectionLoader():
    def __init__(self, input_source, cfg, opt, queueSize=128):
        self.cfg = cfg
        self.opt = opt
        self.input_source = input_source

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                self, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False)

        # Load data file's paths
        file_paths = self.load_data_paths()
        self.all_imgs = defaultdict(list)  # (partial(np.ndarray, 0))
        self.all_boxes = defaultdict(list)
        self.all_scores = defaultdict(list)
        self.all_ids = defaultdict(list)
        self.all_imgs_frame_paths = defaultdict(set)

        # self.all_imgs_frame_paths = {}
        # self.all_imgs = []  # (partial(np.ndarray, 0))
        # self.all_boxes = {}
        # self.all_scores = {}
        # self.all_ids = {}
        # Loop through TFRecord fukes
        for frame_path in tqdm(file_paths):
            # Unpack dataset frin TFRecord file
            dataset = tf.data.TFRecordDataset(frame_path, compression_type='')
            for data in tqdm(dataset):
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                # Group labels of detected objects on a single frame e.g. vehicle, pedestrian etc.
                labels = keypoint_data.group_object_labels(frame)
                for label in labels:
                    # If the detected object is pedestrian or cyclist and the object is detected with a camera
                    if labels[label].object_type == label_pb2.Label.TYPE_PEDESTRIAN or \
                            labels[label].object_type == label_pb2.Label.TYPE_CYCLIST:

                        # Iterate over cameras --> produces same 3D points, but different 2D annotations
                        if labels[label].camera:
                            for cam in labels[label].camera:
                                x = labels[label].camera[cam].box.center_x
                                y = labels[label].camera[cam].box.center_y
                                w = labels[label].camera[cam].box.width
                                h = labels[label].camera[cam].box.length
                                # image_id = str(int(x)) + str(int(y)) + '+' + str(cam) + '_' + label
                                image_id = str(cam) + '_' + label + '.png'

                                #bbox = [x, y, w, h]
                                bbox = [x, y, x+w, y+h]
                                score = 1

                                self.all_boxes[image_id].append(bbox)
                                self.all_scores[image_id].append(score)
                                self.all_ids[image_id].append(0)
                                self.all_imgs_frame_paths[image_id].add(frame_path)

                                # initialize the queue used to store data
                                """
                                pose_queue: the buffer storing post-processed cropped human image for pose estimation
                                """
                                if opt.sp:
                                    self._stopped = False
                                    self.pose_queue = Queue(maxsize=queueSize)
                                else:
                                    self._stopped = mp.Value('b', False)
                                    self.pose_queue = mp.Queue(maxsize=queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        image_preprocess_worker = self.start_worker(self.get_detection)
        return [image_preprocess_worker]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        if self.opt.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        if not self.stopped:
            queue.put(item)

    def wait_and_get(self, queue):
        if not self.stopped:
            return queue.get()

    def get_detection(self):
        # for im_name_k, _ in self.all_imgs_frame_paths.items():
        for im_name_k, _ in self.all_boxes.items():
            boxes = torch.from_numpy(np.array(self.all_boxes[im_name_k]))
            scores = torch.from_numpy(np.array(self.all_scores[im_name_k]))
            ids = torch.from_numpy(np.array(self.all_ids[im_name_k]))

            frame_path = self.all_imgs_frame_paths[im_name_k]
            # cam, label = im_name_k.split('+')[1][:1], im_name_k.split('+')[1][2:]  # im_name_k.split('_')
            cam, label = im_name_k[:1], im_name_k[2:]


            dataset = tf.data.TFRecordDataset(list(frame_path), compression_type='')
            for data in dataset:
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                # Group labels of detected objects on a single frame e.g. vehicle, pedestrian etc.
                labels = keypoint_data.group_object_labels(frame)
                if label in labels:
                    break

            orig_img_k = self.get_image_arr(frame, int(cam))
            #cv2.imwrite("/lhome/rauysal/AlphaPose/results/img/" + im_name_k + '.png', orig_img_k)


            # !!! self._input_size yerine 1000x1000 kullaniyorum
            inps = torch.zeros(boxes.size(0), 3, *self._input_size)
            cropped_boxes = torch.zeros(boxes.size(0), 4)
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img_k, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)
 
                #cropped_img_k = self.get_cropped_image(orig_img_k, box).transpose(2, 0, 1)
                #zeros = np.zeros((3, 1000, 1000))
                #r, c = 0, 0
                #zeros[:, r:r+cropped_img_k.shape[1], c:c+cropped_img_k.shape[2]] += cropped_img_k
                #cropped_img_k = zeros

                #image_tensor = self.image_to_tensor(cropped_img_k)
                #print(image_tensor.shape)
                #inps[i] = image_tensor #torch.reshape(image_tensor, self._input_size)
                #cropped_boxes[i] = torch.FloatTensor(box.float())
                #cv2.imwrite("/lhome/rauysal/AlphaPose/results/vis/custom_crop" + im_name_k + '.png', cropped_img_k)

                #from torchvision.utils import save_image
                #save_image(inps[i], "/lhome/rauysal/AlphaPose/results/vis/custom_crop/test_transform" + im_name_k + '.png')

                #img = tensor_to_image(inps[i])
                #img.save("/lhome/rauysal/AlphaPose/results/vis/custom_crop/test_transform" + im_name_k + '.png')
            self.wait_and_put(self.pose_queue, (inps, orig_img_k, im_name_k, boxes, scores, ids, cropped_boxes))

        self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))
        return

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value

    @property
    def length(self):
        return len(self.all_boxes)

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

    def load_data_paths(self, split_dir=['training/', 'validation/', 'test/']):
        """
        Loads data file's paths from given source
        directory and save them into a list.

        Arguments:
            source_dir (str) -- List storing source directory names

        Returns:
            file_paths (list) -- List storing data files' paths
        """

        file_paths = []

        # Loop through data directories
        for dir in split_dir:
            print(f"Extracting {dir[0:-1]} data")
            data_count = 0
            for root, dirs, files in os.walk(os.path.join(self.input_source, dir)):
                for file in files:
                    if file.endswith('.tfrecord'):
                        # Get all TFRecord files
                        path = str(root) + '/' + str(file)
                        file_paths.append(path)
                        data_count += 1
            print(f"Extracting {dir[0:-1]} data is completed")
            print(f"There are {data_count} TFRecord file for {dir[0:-1]}")
            print("============================================================================================================")
        print(f"There are {len(file_paths)} TFRecord file in total")
        return file_paths

    def _imdecode(self, buf: bytes) -> np.ndarray:
        """
        Function from waymo open dataset examples
        -> https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_keypoints.ipynb

        Receives a binary representation of an image and returns encoded numpy array of the same image.

        args:
            buf -- byte representation the image

        returns:
            np.array(pil) -- array representation of the image
        """
        with io.BytesIO(buf) as fd:
            pil = PIL.Image.open(fd)
            return np.array(pil)

    def get_image_arr(self, frame, cam):
        """_summary_

        Args:
            frame (open dataset frame) Frame of the tfrecord file.
            labels (waymo object labels): Labels of the pedestrain.
            cam (int): Camera in which the data can be found.

        Returns:
            tuple: Croped image and new keypoint coordinates.
        """

        # Get camera images by name
        camera_image_by_name = {i.name: i.image for i in frame.images}
        image_arr = self._imdecode(camera_image_by_name[cam])
        return image_arr

    def get_cropped_image(self, image, bbox, margin=0):
        """_summary_

        Args:
            frame (open dataset frame) Frame of the tfrecord file.
            labels (waymo object labels): Labels of the pedestrain.
            cam (int): Camera in which the data can be found.

        Returns:
            tuple: Croped image and new keypoint coordinates.
        """
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        crop_width = (1 + margin) * h
        crop_height = (1 + margin) * w
        min_x = max(0, int(x - crop_width / 2))
        min_y = max(0, int(y - crop_height / 2))
        max_x = min(image.shape[1] - 1, int(x + crop_width / 2))
        max_y = min(image.shape[0] - 1, int(y + crop_height / 2))
        cropped_image = image[min_y:max_y, min_x:max_x, :]

        return cropped_image


    def image_to_tensor(self, image):
        """Transform ndarray image to torch tensor.
        Parameters
        ----------
        img: numpy.ndarray
            An ndarray with shape: `(H, W, 3)`.
        Returns
        -------
        torch.Tensor
            A tensor with shape: `(3, H, W)`.
        """
        #image = np.transpose(image, (2, 0, 1))  # C*H*W
        image = self.to_torch(image).float()
        if image.max() > 1:
            image /= 255
        return image

    def tensor_to_image(self, tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def to_torch(self, ndarray):
        # numpy.ndarray => torch.Tensor
        if type(ndarray).__module__ == 'numpy':
            return torch.from_numpy(ndarray)
        elif not torch.is_tensor(ndarray):
            raise ValueError("Cannot convert {} to torch tensor"
                            .format(type(ndarray)))
        return ndarray

    def pad_with(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
