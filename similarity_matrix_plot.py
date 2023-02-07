import numpy as np

from CKA_cuda import get_cka

def calculate_similarity_matrix(device, hook_forward, similarity_matrix, aggregate_func='avg'):
    target_layers = [l for l in range(0, 11, 7)]
    cka = get_cka(device)

    for i, layer_ind_i in enumerate(target_layers):
        for j, layer_ind_j in enumerate(target_layers):
            if i > j:
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

def plot_similarity_matrix(similarity_matrix):
    import matplotlib.pyplot as plt

    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            if i > j:
                continue

            plt.matshow(similarity_matrix[i][j])
            plt.title("Maximum CKA similarity of source layer - " + str(i) + "target layer - " + str(j))
            plt.xlabel("target layer patches")
            plt.ylabel("source layer patches")
            plt.colorbar()
            plt.show()
    