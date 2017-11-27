from clize import run
from skimage.io import imsave
import numpy as np
from torch.nn.init import xavier_uniform
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)

colors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 0]
]
colors = np.array(colors, dtype='float32') * 255


class Model(nn.Module):

    def __init__(self, hidden=64, vocab_size=32*32, emb_size=128):
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.init_rnn = nn.Sequential(
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(True),
            nn.Linear(512, hidden)
        )
        self.rnn = nn.RNN(emb_size, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, X, T):
        x = self.features(X)
        x = x.view(x.size(0), -1)
        h = self.init_rnn(x)
        h = h.view(1, h.size(0), h.size(1))
        t = self.emb(T)
        o, _ = self.rnn(t, h)
        o = o.contiguous()
        o = o.view(o.size(0) * o.size(1), self.hidden)
        o = self.out(o)
        return o

    def pred_next(self, x, h):
        x = self.emb(x)
        _, h = self.rnn(x, h)
        o = self.out(h[-1])
        return o, h


def retrieve_col(col, rng=np.random):
    if col == 'random':
        col = colors[rng.randint(0, len(colors) - 1)]
    elif col == 'random_unif':
        col = np.random.randint(0, 255, size=3)
    elif type(col) == int:
        col = colors[col]
    col = np.array(col)
    return col


def generate(nb=100, w=32, h=32, 
             ph=(1, 5), pw=(1, 5), 
             x=(0, 32 -5),y=(0, 32-5),
             nb_patches=(1, 4), random_state=None, fg_color=None,
             bg_color=None, colored=False):
    nb_cols = 3 if colored else 1
    if not bg_color:
        bg_color = [0] * nb_cols
    if not fg_color:
        fg_color = [255] * nb_cols
    rng = np.random.RandomState(random_state)
    X = []
    Y = []
    for _ in range(nb):
        bg_color_ = retrieve_col(bg_color, rng=rng)
        img = np.ones((nb_cols, h, w)) * bg_color_
        label = []
        nb_patches_ = rng.randint(*nb_patches)
        for _ in range(nb_patches_):
            ph_ = rng.randint(*ph)
            pw_ = rng.randint(*pw)
            x_ = rng.randint(*x)
            y_ = rng.randint(*y)
            fg_color_ = retrieve_col(fg_color, rng=rng)
            img[:, y_:y_ + pw_, x_:x_ + ph_] = fg_color_
            label.append((x_, y_, pw_, ph_))
        X.append(img)
        Y.append(label)
    X = np.array(X)
    return X, Y

def train():
    nb = 10000
    ratio_train = 0.9

    epochs = 10000
    batch_size = 128
    w = 32
    h = 32
    px = (0, w-5)
    py = (0, h-5)
    ph = (4, 5)
    pw = (4, 5)
    nb_patches = (1, 4)
    
    X, Y = generate(nb=nb, w=w, h=h, x=px, y=py, ph=ph, pw=pw, nb_patches=nb_patches, random_state=40)
    X = X / 255.0
    H = [hash(tuple(y)) for y in Y]
    exist = set()
    is_dup = np.zeros((len(X),)).astype('bool')
    for i, h in enumerate(H):
        if h not in exist:
            exist.add(h)
            is_dup[i] = False
        else:
            is_dup[i] = True
    X = X[~is_dup]
    Y = [y for d, y in zip(is_dup, Y) if not d]
    
    nb_train = int(len(X) * ratio_train)
    X_train = X[0:nb_train]
    Y_train = Y[0:nb_train]
    X_test = X[nb_train:]
    Y_test = Y[nb_train:]
    print(X_train.shape, X_test.shape)
    transf = {}
    inv_transf = {}
    for i, v in enumerate(range(w)):
        transf[v] = i
        inv_transf[i] = v
    transf['null'] = i + 1
    inv_transf[i + 1] = 'null'

    model = Model(hidden=128, vocab_size=len(transf))
    model = model.cuda()
    model.transf = transf
    model.inv_transf = inv_transf
    model.apply(weights_init)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    avg_precision = 0.0
    avg_loss = 0.0
    nb_updates = 0

    for epoch in range(epochs):
        print('Training...')
        for i in range(0, len(X_train), batch_size):
            x = X_train[i:i + batch_size]
            y = Y_train[i:i + batch_size]
            xlist = []
            ylist = []
            for xi, yi in zip(x, y):
               xi = np.tile(xi, (len(yi), 1, 1, 1))
               xlist.append(xi)
               for p in yi:
                   p = tuple(['null']) + p
                   p = [transf[v] for v in p]
                   ylist.append(p)
            x = np.concatenate(xlist, axis=0)
            y = np.array(ylist)
            x = torch.from_numpy(x).float()
            x = Variable(x)
            x = x.cuda()
            
            y = torch.from_numpy(y).long()
            y = Variable(y).long()
            y = y.cuda()

            tcur  = y[:, 0:-1]
            tnext = y[:, 1:]
            tnext = tnext.contiguous()
            tnext = tnext.view(-1)
            optim.zero_grad()
            tnext_pred = model(x, tcur)
            loss = crit(tnext_pred, tnext)
            loss.backward()
            optim.step()
            _, pred = tnext_pred.max(1)
            precision = (pred==tnext).float().data.mean()
            avg_precision = avg_precision * 0.99  + precision *  0.01
            avg_loss = avg_loss * 0.99 + loss.data[0] * 0.01
            if nb_updates % 100 == 0:
                print('AvgPrecision : {:.5f}, AvgLoss : {:.5f}'.format(avg_precision, avg_loss))
            nb_updates += 1
        print('Epoch {} finished'.format(epoch))
        if epoch % 10 == 0:
            print('Testing...')
            torch.save(model, 'model.th')
            acc_train = _test(model, X_train[0:1000], Y_train[0:1000])
            acc_test = _test(model, X_test, Y_test)
            print('IOU Train : {:.5f}. IOU Test : {:.5f}'.format(acc_train, acc_test))


def _test(model, x, y):
    x = torch.from_numpy(x).float()
    x = Variable(x)
    x = x.cuda()
    h = model.features(x)
    h = h.view(h.size(0), -1)
    h = model.init_rnn(h)
    x = x.data.cpu().numpy().astype('int32')[:, 0]
    acc_mean = []
    for i, (xi, yi, hi) in enumerate(zip(x, y, h)):
        nb = 64
        hi = hi.view(1, 1, -1)
        hi = hi.repeat(1, nb, 1)
        t = torch.zeros(nb, 1)
        t = Variable(t).long()
        t = t.cuda()
        ypredi = [tuple([]) for _ in range(nb)]
        ypredi = torch.zeros(nb, 4)
        for j in range(4):
            o, hi = model.pred_next(t, hi)
            o = o * 2 # temperature
            o = nn.Softmax()(o)
            o = torch.multinomial(o)
            t = o
            ypredi[:, j] = t.data.cpu()
        vals_true = set(yi)
        vals_pred = set([tuple(v) for v in ypredi.tolist()])
        acc = len(vals_pred & vals_true) / len(vals_true | vals_pred)
        acc_mean.append(acc)
    return np.mean(acc_mean)


def test():
    nb_examples = 10
    w = 32
    h = 32
    px = (0, w-5)
    py = (0, h-5)
    ph = (1, 10)
    pw = (1, 10)
    nb_patches = (1, 4)
    X, Y = generate(nb=nb_examples, w=w, h=h, x=px, y=py, ph=ph, pw=pw, nb_patches=nb_patches, random_state=45)
    model = torch.load('model.th')
    x = X
    y = Y
    x = torch.from_numpy(x).float()
    x = Variable(x)
    x = x.cuda()
    ypred = nn.Softmax()(model(x)).data.cpu().numpy()
    x = x.data.cpu().numpy().astype('int32')[:, 0]
    acc_mean = []
    for i, (xi, yi, ypredi) in enumerate(zip(x, y, ypred)):
        xi = xi[:, :, None] * np.ones((1, 1, 3))
        xi[:, :, 0] = 0
        xi[:, :, 2] = 0
        indices = np.argsort(ypredi)[::-1]
        for ind in indices[0:3]:
            x, y, w, h = model.inv_transf[ind]
            xi[y:y + w, x:x + h] += (0, 0, 255)
        xi = xi.astype('int32')
        xi = np.clip(xi, 0, 255)
        imsave('out/out{:02d}.png'.format(i), xi)
        vals = set(np.argsort(ypredi)[::-1][0:len(yi)])
        vals_true = set([model.transf[v] for v in yi])
        acc = len(vals & vals_true) / len(vals_true)
        acc_mean.append(acc)
    print('Mean acc : {:.5f}'.format(np.mean(acc_mean)))
 

if __name__ == '__main__':
    run([train, test])
