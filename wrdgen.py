import torch
import numpy as np
import time
import sys

device = sys.argv[1]

f = open('words.txt')
lines = f.readlines()
f.close()
words = []
frequencies = []
alphabet = set()
for l in lines:
    w,f = l.split(' ')
    f = int(f)
    if(f>50):
        words.append(w)
        frequencies.append(f)
        alphabet = alphabet.union(set(w))
alphabet = list(alphabet)
alphabet.sort()
alphabet.append('#')
print(alphabet)

torch.manual_seed(123)
np.random.seed(234)

class wrdgen(torch.nn.Module):
    def __init__(self,alphabet,n_h):
        super(wrdgen,self).__init__()
        self.alphabet = alphabet
        self.n_h = n_h
        self.lstm = torch.nn.LSTM(input_size = len(alphabet),hidden_size = n_h,num_layers = 10)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_h,n_h),torch.nn.Tanh(),
            torch.nn.Linear(n_h,n_h),torch.nn.Tanh(),
            torch.nn.Linear(n_h,len(alphabet))
        )
        self.h0 = torch.nn.Parameter(torch.randn(10,1,n_h))
        self.c0 = torch.nn.Parameter(torch.randn(10,1,n_h))
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def generate(self,device):
        ret = ''
        with torch.no_grad():
            h = self.h0
            c = self.c0
            inp = torch.zeros(1,1,len(self.alphabet)).to(device)
            inp[0,0,-1] = 1
            while(1):
                output,(h,c) = self.lstm(inp,(h,c))
                output = self.linear(output.view(1,-1)).view(-1).softmax(0).detach().cpu().numpy()
                letter = self.alphabet[np.random.choice(output.shape[0],p=output)]
                if(letter == '#'):
                    break
                ret += letter
        return ret

    def get_loss(self,word,device):
        x = torch.zeros(len(word)+1,1,len(self.alphabet)).to(device)
        x[0,0,-1] = 1
        y = torch.zeros(len(word)+1,dtype=torch.long).to(device)
        y[-1] = len(self.alphabet)-1
        for i in range(len(word)):
            x[i+1,0,self.alphabet.index(word[i])] = 1
            y[i] = self.alphabet.index(word[i])
            
        h,_ = self.lstm(x,(self.h0,self.c0))
        h = h.view(-1,self.n_h)
        h = self.linear(h)
        return self.criterion(h,y)

model = wrdgen(alphabet,500).to(device)
#print(model.generate())

optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

permutation = np.random.permutation(len(words))

start_time = time.time()
for i in range(len(permutation)):
    loss = model.get_loss(words[permutation[i]],device)
    print(i,words[permutation[i]],loss.item(),time.time()-start_time)
    print(model.generate(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if(i%100 == 0):
        torch.save(model.to('cpu'),'wrdgen.pt')
        model.to(device)
