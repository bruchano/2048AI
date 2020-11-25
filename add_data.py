import numpy as np
import random
import torch
import cv2
from PIL import ImageGrab
import requests
import bs4
from lxml import html
from keyboard import mouse
import keyboard
import pyautogui
import time
import pynput.keyboard
import os
from NET import *


path = "training_data_5.pt"

if os.path.isfile(path):
    print("file exist")
    training_data = torch.load(path)
else:
    print("No training data file exist")
    training_data = []


def get_training_data():
    while True:
        if keyboard.is_pressed("up"):
            screen = ImageGrab.grab((680, 340, 1185, 840))
            screen = np.array(screen)[:, :, ::-1]
            key = [1, 0, 0, 0]
            training_data.append([screen, key])

            print("up")
            time.sleep(0.1)

        if keyboard.is_pressed("down"):
            screen = ImageGrab.grab((680, 340, 1185, 840))
            screen = np.array(screen)[:, :, ::-1]
            key = [0, 1, 0, 0]
            training_data.append([screen, key])

            print("down")
            time.sleep(0.1)

        if keyboard.is_pressed("left"):
            screen = ImageGrab.grab((680, 340, 1185, 840))
            screen = np.array(screen)[:, :, ::-1]
            key = [0, 0, 1, 0]
            training_data.append([screen, key])

            print("left")
            time.sleep(0.1)

        if keyboard.is_pressed("right"):
            screen = ImageGrab.grab((680, 340, 1185, 840))
            screen = np.array(screen)[:, :, ::-1]
            key = [0, 0, 0, 1]
            training_data.append([screen, key])

            print("right")
            time.sleep(0.1)

        if len(training_data) % 100 == 0 and len(training_data) != 0:
            print(len(training_data))
            torch.save(training_data, path)
            print("saved")

        if keyboard.is_pressed("q"):
            print(len(training_data))
            torch.save(training_data, path)
            print("saved and break")
            break


get_training_data()
