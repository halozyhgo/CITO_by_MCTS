# N = 200
# sum_1 = 1
# for i in range(1,N+1):
#     sum_1 *= i
# print(sum_1)
import numpy as np
import torch
import torch.nn.functional as F
# logits = [[3.,7.,6.,7.,8.,30,23],[6.,7.,6.,7.,8.,30,23]]
# logits = np.array(logits)
# logits = torch.from_numpy(logits)
# print('logits')
# print(logits)
# pred = F.softmax(logits)
# print('pred')
# print(pred)
# print('pred.numpy()')
# print(pred.numpy())
# print('pred.max(dim=0)[0]')
#
#
# q_value = F.softmax(logits,dim=0)
# print(q_value)
# action  = np.argmax(q_value)
# print(action.item())
print(torch.cuda.is_available())