import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


def plot_policy(policy, mode, title):

    img = Image.open("./gridworld.png")
    fig,ax = plt.subplots()
    ax.imshow(img)
    
    marker_style = {0 : '^', 1 : '>', 2 : 'v', 3 : '<'}
    color_style = ['r', 'b', 'g', 'y']

    end_state = {
            'A' : (0, 11),
            'B' : (2, 9),
            'C' : (6, 7)
        }

    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            
            if end_state[mode] == (i, j):
                continue

            px = img.size[0] * (j + 0.5) / 12 
            py = img.size[1] * (i + 0.5) / 12 
            ax.scatter(px, py, marker = marker_style[policy[i][j]], c = color_style[policy[i][j]], s = 60)
        
    plt.axis('off')
    plt.title(title + "- Optimal Policy for experiment - " + mode, fontsize = 24)
    plt.show()
    