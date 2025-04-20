import torch
import torch.nn as nn

from board3 import Board3
from controller3 import ActionController


def to_emb(board: Board3):

    py, px = board.get_player_position()
    ey, ex = board.get_enemy_position()
    ty, tx = board.get_todd_position()

    mws = [0] * 16

    for (y, x) in board.mw:
        mws[y * 4 + x] = 1

    pe = torch.cat([
        nn.functional.one_hot(torch.LongTensor([py]), num_classes=4),
        nn.functional.one_hot(torch.LongTensor([px]), num_classes=4),
        nn.functional.one_hot(torch.LongTensor([ey]), num_classes=4),
        nn.functional.one_hot(torch.LongTensor([ex]), num_classes=4),
        nn.functional.one_hot(torch.LongTensor([ty]), num_classes=4),
        nn.functional.one_hot(torch.LongTensor([tx]), num_classes=4)
      ], dim=0)

    mwe = torch.Tensor([mws])


    return torch.cat([pe.flatten().unsqueeze(0).float(), mwe], dim=1)

def emb_mem(mem, nc=2):
    fe = []
    for b, a in mem:
        e1 = to_emb(b)
        e2 = nn.functional.one_hot(torch.LongTensor([a]), num_classes=nc)
        fe.append(torch.cat([e1, e2], dim=1))

    return torch.cat(fe, dim=0).unsqueeze(0)


def gather_history(hist, c=3):
    res = []
    for i in range(len(hist)):
        line = []
        for j in range(c):
            if i - j < 0:
                break
            line.append(hist[i - j])
        if len(line) < c:
            for _ in range(c - len(line)):
                line.append((line[-1][0], 0))
        res.append(line[::-1])

    return res

def emb_mem(mem):
    hl = []
    for b, a in mem:
        board_embedding = to_emb(b)
        action_embedding = nn.functional.one_hot(torch.LongTensor([a]), ActionController(b).get_action_space())
        hl.append(torch.cat([board_embedding, action_embedding], dim=1))
    return torch.cat(hl, dim=0).unsqueeze(0)

#
# class NNT:
#
#     def __init__(self, board, player, ml=2, step=1600, walk_time=500, mem=None):
#         self.ml = ml
#         self.step = step
#         self.walk_time = walk_time
#         self.board = board
#         self.controller = ActionController(board)
#         self.children = None
#         self.player = player
#         if mem is not None:
#             self.mem = deque(mem, maxlen=ml)
#         else:
#             self.mem = deque([(board, 0)] * ml, maxlen=ml)
#         self.a = None
#         self.p = 0
#         self.v = 0
#         self.n = 0
#         pass
#
#
#     def get_winner(self):
#         if self.controller.is_win():
#             return 1
#         if self.controller.is_lose():
#             return -1
#         if self.controller.is_block():
#             return -1
#
#         return None
#
#     def search(self, fp):
#
#         winner = self.get_winner()
#         if winner is not None:
#             self.n += 1
#             self.v = winner
#             return self.v
#
#
#         if self.children is None:
#             self.expand(fp)
#
#             return self.v
#
#         cs = sum([x.n for x in self.children])
#         sv = max(self.children, key=lambda x: x.v + 1.5 * x.p * math.sqrt(cs / (x.n + 1)))
#         res = sv.search(fp)
#         self.v = (self.v * self.n + res) / (self.n + 1)
#         self.n += 1
#
#         return res
#
#     def expand(self, fp):
#         self.children = []
#         vs = []
#
#         with torch.no_grad():
#             pl, vl = fp(to_emb(self.board), emb_mem(self.mem, ActionController.get_action_space()))
#             plv = pl.exp().squeeze(0)
#
#
#         for p in self.controller.get_available_moves():
#             board_copy = self.board.copy()
#             nt = NNT(board_copy, -self.player, ml=self.ml, step=self.step, walk_time=self.walk_time, mem=list(self.mem))
#             nt.controller.execute_action(p)
#             board_copy.step(nt.step, walk_time=nt.walk_time)
#             nt.mem.append((self.board, p))
#             nt.a = p
#
#             winner = nt.get_winner()
#
#             nt.v = 0 if winner is None else winner
#             nt.p = plv[p].item()
#             nt.n = 1
#             vs.append(nt.p)
#             # board_copy.swap_enemy()
#             self.children.append(nt)
#
#         self.v = vl.item()
#         self.n = len(vs)
#
#     def norm_pol(self):
#         v = [0] * self.controller.get_action_space()
#         for c in self.children:
#             v[c.a] = c.n
#
#         return [x / sum(v) for x in v]
#
#
#
# def play(fp, bb, step=1600, max_moves=40, iterc=200, ml=2, use_best=False):
#     fp.eval()
#     b = bb()
#
#     controller = ActionController(b)
#     nt = NNT(b, 1, ml=2, step=step, walk_time=1000)
#
#     mem = deque([(b, 0)] * ml, maxlen=ml)
#     hist = []
#     for i in range(max_moves):
#
#         bc = b.copy()
#
#         if use_best:
#             with torch.no_grad():
#                 out, _ = fp(to_emb(b), emb_mem(mem, controller.get_action_space()))
#                 a = out.exp().argmax().item()
#         else:
#             for _ in range(iterc):
#                 nt.search(fp)
#             mc = max(nt.children, key=lambda x:x.n)
#             a = mc.a
#
#             hist.append((bc, list(mem), nt.norm_pol()))
#
#         mem.append((bc, a))
#
#         controller.execute_action(a)
#         b.step(step)
#
#         nt = NNT(bc, 1, ml=2, step=step, walk_time=1000)
#
#         if controller.is_win():
#             return hist, 1
#         if controller.is_lose():
#             return hist, -1
#         if controller.is_block():
#             return hist, -1
#
#
#
#     return hist, -1
#
#

#
# def search_play(fp, bb, step=1600, max_moves=40, iterc=200, ml=2, use_best=False):
#     b = bb()
#     path = search(b, 100000, time_step=step)
#
#     if path is None:
#         raise Exception('no path')
#
#     return [(b, m, a) for (b, a), m in zip(path, gather_history(path, ml))], 1
#
#
#
#
# def gen_data(fp, bb, n=1000):
#     boards = []
#     memories = []
#     actions = []
#     rewards = []
#     while len(boards) < n:
#         h, r = search_play(fp, bb)
#         bds, msm, acts = list(zip(*h))
#         boards.extend([to_emb(b) for b in bds])
#         memories.extend([emb_mem(m, ActionController.get_action_space()) for m in msm])
#         # actions.extend(torch.Tensor([acts]))
#         actions.extend(acts)
#         rewards.extend([r] * len(h))
#
#     # return TensorDataset(torch.cat(boards, dim=0), torch.cat(memories, dim=0), torch.cat(actions, dim=0), torch.Tensor(rewards))
#     return TensorDataset(torch.cat(boards, dim=0), torch.cat(memories, dim=0), torch.LongTensor(actions), torch.Tensor(rewards))
#
# def eval_fc(fc, bb, ng=20):
#     fc.eval()
#     w = 0
#     l = 0
#     for _ in range(ng):
#         h, r = play(fc, bb, use_best=True)
#         if r == 1:
#             w += 1
#         else:
#             l += 1
#
#     return w / ng, l / ng
#
# def train_nn(fp, dataset, epochs=200, lr=0.001):
#
#     crit1 = nn.CrossEntropyLoss()
#     crit2 = nn.MSELoss()
#     opt = optim.SGD(fp.parameters(), lr=lr)
#
#
#     lh = []
#     for e in range(epochs):
#         el = []
#         fp.train()
#         for b, m, a, r in DataLoader(dataset, batch_size=32, shuffle=True):
#             opt.zero_grad()
#             y, v = fp(b, m)
#             # l = crit1(y, a) + crit2(v.squeeze(1), r)
#             l = torch.sum(-torch.gather(y, 1, a.unsqueeze(1)) * v.detach()) + crit2(v.squeeze(1), r)
#             l.backward()
#             opt.step()
#             el.append(l.item())
#         lh.append(sum(el) / len(el))
#
#     return lh
#
#
#
# fp = FP(to_emb(Board3()).shape[1], ActionController.get_action_space())
#
# def buildb_lvl_0():
#     b = Board3(walk_time=1000)
#
#     b.players_positions = [(3, 0), (0, 0), (3, 3)]
#     b.set_todd(*random.choice(list(set(tod_cells) - {(3, 0), (0, 0)})))
#     return b
#
# def buildb_lvl_1():
#     b = Board3(walk_time=1000)
#
#     b.players_positions = [(3, 0), (0, 0), (3, 3)]
#     b.set_todd(*random.choice(list(set(tod_cells) - {(3, 0), (0, 0)})))
#     b.set_player(*random.choice(list(set(empty_cells) - {b.get_todd_position(), (0, 0)})))
#     return b
#
# def buildb_lvl_2():
#     b = Board3(walk_time=1000)
#     ey, ex = random.choice(list([(0, 0), (0, 2)]))
#     ty, tx = random.choice(list(set(tod_cells) - {(ey, ex)}))
#     py, px = random.choice(list(set(empty_cells) - {(ty, tx), (ey, ex)}))
#     b.players_positions = [(py, px), (ey, ex), (ty, tx)]
#     return b
#
# def buildb_lvl_3():
#     b = Board3(walk_time=1000)
#     ey, ex = random.choice(list([(0, 0), (0, 2), (1, 0), (1, 2)]))
#     ty, tx = random.choice(list(set(tod_cells) - {(ey, ex)}))
#     py, px = random.choice(list(set(empty_cells) - {(ty, tx), (ey, ex)}))
#     b.players_positions = [(py, px), (ey, ex), (ty, tx)]
#     return b
#
# def buildb_lvl_4():
#     b = Board3(walk_time=1000)
#     ey, ex = random.choice(list([(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]))
#     ty, tx = random.choice(list(set(tod_cells) - {(ey, ex)}))
#     py, px = random.choice(list(set(empty_cells) - {(ty, tx), (ey, ex)}))
#     b.players_positions = [(py, px), (ey, ex), (ty, tx)]
#     return b
#
# def buildb_lvl_5():
#     b = Board3(walk_time=1000)
#     return b
#
#
#
#
# build_functions = [
#     buildb_lvl_0,
#     buildb_lvl_1,
#     buildb_lvl_2,
#     buildb_lvl_3,
#     buildb_lvl_4,
#     buildb_lvl_5,
#
# ]
#
# current_lvl = 0
#
# wh = []
# for _ in tqdm(range(200)):
#
#     dt = gen_data(fp, build_functions[current_lvl], 32)
#
#     loss_hist = train_nn(fp, dt, epochs=5, lr=0.00001)
#     wn, ls = eval_fc(fp, build_functions[current_lvl], 40)
#     wh.append(wn)
#
#     ws = []
#     for p in fp.parameters():
#         ws.append(p.data.norm(2).item())
#
#     if wn > 0.85 and current_lvl < len(build_functions) - 1:
#         current_lvl += 1
#         print(f'lvl increase {current_lvl}')
#
#
#
# plt.plot(wh, c='g')
# plt.show()



# class MHAttention(nn.Module):
#     def __init__(self, n_dim=64, n_heads=2):
#         super(MHAttention, self).__init__()
#         self.n_dim = n_dim
#         self.n_heads = n_heads
#         self.neg_fill = -1e4
#         self.lq = nn.Linear(n_dim, n_dim)
#         self.lk = nn.Linear(n_dim, n_dim)
#         self.lv = nn.Linear(n_dim, n_dim)
#         self.dropout = nn.Dropout(0.1)
#
#         self.lc = nn.Linear(n_dim, n_dim)
#
#
#     def forward(self, q, k, v, mask=None):
#         wq = self._head_reshape(self.lq(q))
#         wk = self._head_reshape(self.lk(k))
#         wv = self._head_reshape(self.lv(v))
#
#         a = wq @ wk.transpose(2, 3) / math.sqrt(self.n_dim)
#         if mask is not None:
#             a = a.masked_fill(mask.unsqueeze(1), self.neg_fill)
#         a = torch.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
#         a = self.dropout(a)
#
#         o = a @ wv
#         return self.lc(self._head_reshape_back(o))
#
#     def _head_reshape(self, x):
#         return x.view(x.shape[0], x.shape[1], self.n_heads, self.n_dim // self.n_heads).transpose(1, 2)
#
#     def _head_reshape_back(self, x):
#         return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.n_dim)
#
# class EncoderLayer(nn.Module):
#     def __init__(self, n_dim=64, n_heads=2, ff_dim=128):
#         super(EncoderLayer, self).__init__()
#         self.mha = MHAttention(n_dim, n_heads)
#         self.l1 = nn.LayerNorm(n_dim, )
#
#         self.ffw = nn.Sequential(
#             nn.Linear(n_dim, ff_dim, ),
#             nn.ReLU(),
#             nn.Linear(ff_dim, n_dim, ),
#             nn.Dropout(0.1)
#         )
#         self.l2 = nn.LayerNorm(n_dim, )
#
#     def forward(self, x):
#         ao = self.mha(x, x, x)
#         h = self.l1(x + ao)
#
#         fo = self.ffw(h)
#         return self.l2(h + fo)
#
#
# class Encoder(nn.Module):
#     def __init__(self, num_layers=2, n_dim=64, n_heads=2, ff_dim=128):
#         super(Encoder, self).__init__()
#         self.layers = nn.ModuleList([EncoderLayer(n_dim, n_heads, ff_dim) for _ in range(num_layers)])
#
#     def forward(self, x):
#         for l in self.layers:
#             x = l(x)
#         return x
#
#
# class FP(nn.Module):
#     def __init__(self, n_dim, a_space):
#         super(FP, self).__init__()
#         self.n_dim = n_dim
#         self.a_space = a_space
#         self.fc = nn.Sequential(
#             nn.Linear(n_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         self.encoder = Encoder(num_layers=1, n_dim=64, n_heads=4, ff_dim=128)
#
#         # self.hf = nn.Sequential(
#         #     nn.Flatten(start_dim=1),
#         #     nn.BatchNorm1d(120),
#         #     nn.Linear(120, 128),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.1),
#         #     nn.Linear(128, 256),
#         # )
#
#
#         self.fp = nn.Sequential(
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#
#             nn.Linear(128, a_space),
#             nn.Softmax(dim=-1)
#         )
#
#         self.fv = nn.Sequential(
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#
#             nn.Linear(128, 1),
#         )
#
#     def forward(self, x):
#         e = self.fc(x)
#         hidden = self.encoder(e.unsqueeze(1)).squeeze(1)
#         return self.fp(hidden), self.fv(hidden)
#

