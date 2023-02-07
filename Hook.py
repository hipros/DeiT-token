import torch

class Hook():
    cls_array = []
    def __init__(self, module, backward=False):
        self.debug=False

        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input[0]
        self.output = output[0]
        self.token_input = self.generate_token_input()
        self.input_gram = self.generate_gram()
        

        if self.debug:
            print("---------------")
            print(f'src: {input[0].shape}')
            print(f'tgt: {output[0].shape}')
            print(f'gram: {self.gram.shape}')
            print("---------------")

    def generate_token_input(self):
        # (mb, p, d) -> (p, mb, d)
        token_input = self.input.permute(1, 0, 2)

        if self.debug:
            print("input shape (mb, p, d): ", self.input.shape)
            print("origin: ", token_input.shape)

        return token_input

    def generate_gram(self):
        # (p, mb, d) -> (p, d, mb)
        trans = self.token_input.permute(0, 2, 1)

        # (p, mb, d), (p, d, mb) -> (p, mb, mb)
        gram = torch.matmul(self.token_input, trans)

        if self.debug:
            print("input shape (mb, p, d): ", self.input.shape)
            print("origin: ", self.token_input.shape)
            print("origin^T: ", trans.shape)

            print("gram: ", gram.shape)

            print("one feature: ", self.input[0].shape)
            print("one gram: ", gram[0].shape)

        return gram

    def close(self):
        self.hook.remove()
        