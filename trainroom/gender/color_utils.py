import numpy as np
from skimage import color

def hex_to_bgr(hex_rgb):
    h = hex_rgb.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
    return rgb[2], rgb[1], rgb[0]

def get_skin_colors_old():
    color_map = {}
    color_map[0] = '#f7dfd1'
    color_map[1] = '#f5c4ac'
    color_map[2] = '#f6ad8f'
    color_map[3] = '#cd866c'
    color_map[4] = '#f5a779'
    color_map[5] = '#c48257'
    color_map[6] = '#a97457'
    color_map[7] = '#925e44'
    color_map[8] = '#6f4b34'
    color_map[9] = '#4d372d'
    return color_map

def get_skin_colors():
    color_map = {}
    color_map[0] = '#f0d6cc'
    color_map[1] = '#e7bead'
    color_map[2] = '#dfb49e'
    color_map[3] = '#f1c8b1'
    color_map[4] = '#e0ae96'
    color_map[5] = '#d39a7c'
    color_map[6] = '#dea48d'
    color_map[7] = '#d29379'
    color_map[8] = '#b47d66'
    color_map[9] = '#a36346'
    color_map[10] = '#895540'
    color_map[11] = '#412a25'
    return color_map

def get_race_colors():
    color_map = {}
    color_map[0] = '#f7dfd1'
    color_map[1] = '#925e44'
    color_map[2] = '#c48257'
    color_map[3] = '#f5c4ac'
    return color_map

def get_eyes_colors_old():
    color_map = {}
    color_map[0] = '#dad1cb'
    color_map[1] = '#9c9591'
    color_map[2] = '#422918'
    color_map[3] = '#7c4422'
    color_map[4] = '#b17f49'
    color_map[5] = '#338b56'
    color_map[6] = '#a1bd56'
    color_map[7] = '#79a5ad'
    color_map[8] = '#8fc5cf'
    color_map[9] = '#3974a4'
    color_map[10] = '#33414d'
    color_map[11] = '#88712d'
    return color_map

def get_eyes_colors():
    color_map = {}
    color_map[0] = '#9c9591'
    color_map[1] = '#422918'
    color_map[2] = '#7c4422'
    color_map[3] = '#338b56'
    color_map[4] = '#79a5ad'
    color_map[5] = '#33414d'
    color_map[6] = '#88712d'
    return color_map

def get_lips_colors_old():
    color_map = {}
    color_map[0] = '#edb0b3'
    color_map[1] = '#d58585'
    color_map[2] = '#de7973'
    color_map[3] = '#9e584f'
    color_map[4] = '#f0836c'
    color_map[5] = '#914940'
    color_map[6] = '#8d4941'
    color_map[7] = '#844239'
    color_map[8] = '#67332b'
    color_map[9] = '#411f1e'
    color_map[10] = '#aa0613'
    color_map[11] = '#e64f51'
    return color_map

def get_lips_colors():
    color_map = {}
    color_map[0] = '#d58585'
    color_map[1] = '#de7973'
    color_map[2] = '#9e584f'
    color_map[3] = '#67332b'
    color_map[4] = '#e64f51'
    return color_map

def get_brows_colors():
    color_map = {}
    color_map[0] = '#222428'
    color_map[1] = '#392921'
    color_map[2] = '#59463c'
    color_map[3] = '#776452'
    color_map[4] = '#7a5744'
    color_map[5] = '#af9575'
    color_map[6] = '#ffeeda'
    color_map[7] = '#843d2a'
    color_map[8] = '#d5702f'
    color_map[9] = '#867d7e'
    color_map[10] = '#c2b1a8'
    return color_map

def get_hair_colors_old():
    color_map = {}
    color_map[0] = '#000000'
    color_map[1] = '#30211a'
    color_map[2] = '#3c1214'
    color_map[3] = '#5e4130'
    color_map[4] = '#915b32'
    color_map[5] = '#936851'
    color_map[6] = '#e5bc8a'
    color_map[7] = '#fdf1e1'
    color_map[8] = '#a99d9e'
    color_map[9] = '#e6d4ca'
    color_map[10] = '#d5702f'
    return color_map

def get_hair_colors_old2():
    color_map = {}
    color_map[0] = '#000000'
    color_map[1] = '#30211a'
    color_map[2] = '#3c1214'
    color_map[3] = '#5e4130'
    color_map[4] = '#915b32'
    color_map[5] = '#936851'
    color_map[6] = '#e5bc8a'
    color_map[7] = '#fdf1e1'
    color_map[8] = '#d5702f'
    return color_map

def get_hair_colors_old3():
    color_map = {}
    color_map[0] = '#000000'
    color_map[1] = '#30211a'
    color_map[2] = '#5e4130'
    color_map[3] = '#915b32'
    color_map[4] = '#936851'
    color_map[5] = '#e5bc8a'
    color_map[6] = '#d5702f'
    return color_map

def get_hair_colors():
    color_map = {}
    color_map[0] = '#000000' # black
    color_map[1] = '#30211a' # dark brown
    color_map[2] = '#5e4130' # brown
    color_map[3] = '#915b32' # ginger-brown
    color_map[4] = '#bc906e' # fair-haired
    color_map[5] = '#e5bc8a' # blonde
    color_map[6] = '#fdf1e1' # platinum-blonde
    color_map[7] = '#9e9089' # gray-haired
    color_map[8] = '#d5702f' # ginger
    return color_map

def color_distances(color_map):

    color_list = [(k, hex_to_bgr(color_map[k])) for k in color_map]
    color_keys, color_array = zip(*color_list)

    colors_b, colors_g, colors_r = zip(*color_array)

    bgr = np.zeros((1, len(color_array), 3))
    bgr[0, :, 0] = np.array(colors_b) / 255.
    bgr[0, :, 1] = np.array(colors_g) / 255.
    bgr[0, :, 2] = np.array(colors_r) / 255.

    lab_array = color.rgb2lab(bgr)

    colors_L = list(lab_array[0, :, 0])
    colors_a = list(lab_array[0, :, 1])
    colors_b = list(lab_array[0, :, 2])

    lab_array = zip(colors_L, colors_a, colors_b)

    lab_list = zip(color_keys, list(lab_array))

    color_map_lab = {}
    for k, lab_color in lab_list:
        color_map_lab[k] = lab_color

    # L = [0,100]
    # a = [-87,99]
    # b = [-108,95]

    n_colors = len(color_keys)
    dist_mat = {}

    maxdif_L = 100.0 - 0.0
    maxdif_a = 98.2352 + 86.1813
    maxdif_b = 94.4758 + 107.862

    norm_coeff_LAB = np.sqrt(maxdif_L*maxdif_L+maxdif_a*maxdif_a+maxdif_b*maxdif_b)

    for i in range(n_colors):
        for j in range(i, n_colors):

            color_key_i = color_keys[i]
            color_key_j = color_keys[j]

            color_i = color_map_lab[color_key_i]
            color_j = color_map_lab[color_key_j]

            # compute deltaE
            deltaE = np.linalg.norm(np.array(color_i) - np.array(color_j))

            min_i = min(i, j)
            max_j = max(i, j)
            sel_key = '{}_{}'.format(min_i, max_j)

            # dist_mat[sel_key] = deltaE / norm_coeff_LAB
            dist_mat[sel_key] = deltaE

    return dist_mat

def get_delta_e(color_dist, i, j):

    min_i = min(i, j)
    max_j = max(i, j)
    sel_key = '{}_{}'.format(min_i, max_j)

    return color_dist[sel_key]


if __name__ == '__main__':
    # test

    race_colors_hex = get_race_colors()
    print(color_distances(race_colors_hex))