res4-seed*.log
实验环境：
        p_hidden_size = 1024
        l_hidden_size = 512
        hidden_size = 1024
        num_blocks = 32
        网络最后一层softmax，在sample时去掉softmax
        cost = torch.sum(pred * one_hots * rewards)
        cost /= pred.shape[0]
和res-4.md作对比
criterion采用multiprob
reward = loss_i - loss_(i-1)
optimizer = sgd + clip