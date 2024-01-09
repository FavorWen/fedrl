res-100-4-seed*.log
client_nums = 100
participants = 10
history_dim = 8
实验环境：
        p_hidden_size = 1024
        l_hidden_size = 512
        hidden_size = 1024 * 2
        num_blocks = 32 + 16
        网络最后一层softmax，在sample时去掉softmax
        cost = torch.sum(pred * one_hots * rewards)
        cost /= pred.shape[0]
以res-100-4为基准，实验reward变成 lossi - baseline
baseline = avaging windows T 
```python: helper.py
        T = 1
        for idx in range(first, first+hdim):
            log_list.append(self.buffer[idx].encoding(client_nums))
            T += 1
            baseline = baseline * (T-1)/ T + self.buffer[idx].reward()/T
```