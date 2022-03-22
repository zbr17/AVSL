import torch

def pca(A, decenter=True, topk=None):
    # svd
    if decenter:
        mean = torch.mean(A, dim=0, keepdim=True)
        A = A - mean
    cor_mat = torch.matmul(A.t(), A)
    U, S, V = torch.svd(cor_mat)
    # return value
    if topk is None:
        return U.data
    else:
        return U[:, :topk].data

if __name__ == "__main__":
    from torch.autograd import Variable
    A = torch.randn(2000, 128)
    A = Variable(A, requires_grad=True)
    device = torch.device("cuda:0")
    A = A.to(device)
    U = pca(A, topk=10)
    pass