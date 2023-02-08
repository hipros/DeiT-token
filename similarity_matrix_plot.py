import numpy as np
import os

from CKA_cuda import get_cka

def calculate_similarity_matrix(device, hook_forward, similarity_matrix, target_layers, aggregate_func='avg'):
    cka = get_cka(device)

    for i, layer_ind_i in enumerate(target_layers):
        for j, layer_ind_j in enumerate(target_layers):
            if i >= j:
                continue

            print("################# source layer - ", layer_ind_i, ", target layer -" ,layer_ind_j, "#################")

            src_blk_input = hook_forward[layer_ind_i].token_input
            tgt_blk_input = hook_forward[layer_ind_j].token_input

            for ind_i, token_i in enumerate(src_blk_input):
                src_token_cka_max = 0.0
                src_token_cka_sum = 0.0
                for ind_j, token_j in enumerate(tgt_blk_input):
                    token_cka = cka.linear_CKA(token_i, token_j)
                    src_token_cka_max = max(src_token_cka_max, token_cka)
                    src_token_cka_sum += token_cka

                    similarity_matrix[i][j][ind_i][ind_j] += token_cka

def plot_similarity_matrix(similarity_matrix, output_dir, target_layers):
    import matplotlib.pyplot as plt

    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            if i >= j:
                continue

            plt.matshow(similarity_matrix[i][j])
            plt.title("CKA similarity on patches")
            plt.xlabel(str(target_layers[j]) + " layer patches")
            plt.ylabel(str(target_layers[i]) + " layer patches")
            plt.colorbar()
            plt.clim(0.0 , 1.0)
            # plt.show()
            path = os.path.join(output_dir, "CKA_similarity_matrix_layer" + str(target_layers[i]) + "_layer" + str(target_layers[j]) + ".pdf")
            plt.savefig(path, bbox_inches='tight', pad_inches=0.5, )
            plt.clf()
    