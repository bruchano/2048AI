import torch
from torch import optim
from torch.nn import MSELoss, CrossEntropyLoss
import cv2
from PIL import Image, ImageGrab
import numpy as np
from math import *
import pyautogui
import time
import keyboard
from keyboard import mouse
import random
from NET import *

DATA_PATH = "training_data_5.pt"

EPOCH = 3
MODEL = 3
LR = 1e-2
VERSION = 7
SAVE_MODEL_PATH = f"2048AI_model_{MODEL}_lr_{LR}_ver_{VERSION}.pt"

SAVED_MODEL = 3
SAVED_LR = 1e-2
SAVED_VERSION = 7
model_path = f"2048AI_model_{MODEL}_lr_{SAVED_LR}_ver_{SAVED_VERSION}.pt"


def train(MODEL_PATH=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player = AutoPlayer().to(device)
    if MODEL_PATH:
        player.load_state_dict(torch.load(MODEL_PATH))
    player.train()

    # target = AutoPlayer().to(device)
    # target.load_state_dict(player.state_dict())

    training_data = torch.load(DATA_PATH)
    optimizer = optim.Adam(player.parameters(), lr=LR)
    mse_loss = MSELoss()

    epoch = EPOCH
    for e in range(epoch):
        print("--epoch %d--" % (e + 1))
        random.shuffle(training_data)
        loop = 0
        for sample in training_data:
            print("--loop %d--" % loop)

            optimizer.zero_grad()

            state, target = sample
            state = torch.from_numpy(state).transpose(0, 2).unsqueeze(0).to(device) / 255.
            state = state.type(torch.float)

            output = player(state).type(torch.float)
            print("output:", output)
            target = torch.tensor(target).type(torch.float)
            print("target:", target)

            loss = mse_loss(output, target)
            print("loss:", loss.item())
            loss.backward()

            optimizer.step()
            loop += 1

            # screen = ImageGrab.grab((680, 340, 1185, 840))
            # screen = np.array(screen)[:, :, ::-1].copy()
            # state = torch.from_numpy(screen).transpose(0, 2).unsqueeze(0).to(device) / 255.
            # score = ImageGrab.grab((1000, 180, 1045, 200))
            # score = np.array(score).copy()
            #
            # output = player(state)
            # print("output:", output)
            # max_move = torch.argmax(output)
            # print("next move:", max_move.item())
            #
            # next_move = MOVE[max_move]
            # pyautogui.press(next_move)
            #
            # new_screen = ImageGrab.grab((680, 340, 1185, 840))
            # new_screen = np.array(new_screen)[:, :, ::-1].copy()
            # new_state = torch.from_numpy(new_screen).transpose(0, 2).unsqueeze(0).to(device) / 255.
            # new_score = ImageGrab.grab((1000, 180, 1045, 200))
            # new_score = np.array(new_score).copy()
            #
            # next_output = target(new_state)
            # reward = torch.zeros(1, 4)
            # repeat = (new_screen == screen).all()
            # repeat_score = (new_score == score).all()
            #
            # if repeat:
            #     print("score: -3")
            #     reward[0, max_move] -= 3
            # elif not repeat_score:
            #     print("score: +10")
            #     reward[0, max_move] += 5
            # else:
            #     print("score: +0")
            #
            # if list(screen[310, 250]) == [102, 122, 143]:
            #     pyautogui.click(930, 650)
            #     print("--start new game--")
            #     time.sleep(1)
            #     continue
            #
            # expected_value = output
            # actual_value = y * next_output + reward
            #
            # loss = mse_loss(expected_value, actual_value)
            # loss.backward()
            # optimizer.step()
            #
            # loop += 1
            # if loop % 5 == 0:
            #     print("--step--")
            #     target.load_state_dict(player.state_dict())

            if keyboard.is_pressed("q"):
                print("--save break--")
                torch.save(player.state_dict(), SAVE_MODEL_PATH)
                break
            if keyboard.is_pressed("e"):
                print("--no save break--")
                break

    torch.save(player.state_dict(), SAVE_MODEL_PATH)


def evaluate(MODEL_PATH):
    time.sleep(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    player = AutoPlayer().to(device)
    player.load_state_dict(torch.load(MODEL_PATH))
    player.eval()

    t = time.time()
    while True:
        screen = ImageGrab.grab((680, 340, 1185, 840))
        screen = np.array(screen)[:, :, ::-1].copy()
        state = torch.from_numpy(screen).transpose(0, 2).unsqueeze(0).to(device) / 255.

        output = player(state)
        move = torch.argmax(output)
        print("next move:", move.item())
        pyautogui.press(MOVE[move])

        print("time taken: %f" % (time.time() - t))
        t = time.time()

        new_screen = ImageGrab.grab((680, 340, 1185, 840))
        new_screen = np.array(new_screen)[:, :, ::-1].copy()

        if (new_screen == screen).all():
            print("--press down--")
            pyautogui.press(MOVE[1])
            screen = ImageGrab.grab((680, 340, 1185, 840))
            screen = np.array(screen)[:, :, ::-1].copy()
            if (new_screen == screen).all():
                print("--press right--")
                pyautogui.press(MOVE[3])
                new_screen = ImageGrab.grab((680, 340, 1185, 840))
                new_screen = np.array(new_screen)[:, :, ::-1].copy()
                if (new_screen == screen).all():
                    print("--press left--")
                    pyautogui.press(MOVE[2])
                    screen = ImageGrab.grab((680, 340, 1185, 840))
                    screen = np.array(screen)[:, :, ::-1].copy()
                    if (new_screen == screen).all():
                        print("--press up--")
                        pyautogui.press(MOVE[0])

        if keyboard.is_pressed("q"):
            print("--break--")
            break


evaluate(model_path)

print("Done")
