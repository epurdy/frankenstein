from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from frankenstein.utils import get_tokens, get_str_tokens, detect_outliers

import matplotlib.patches as mpatches
import random
import colorsys
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def nice_color(i, n):
#    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    h = (i // 2) / n
    #s = 0.5 + (i % 2) * 0.5
    s = 1
    l = 0.3 + (i % 2) * 0.4
    r,g,b = colorsys.hls_to_rgb(h,l,s)
    return (r, g, b)

text = 'The quick red fox jumps over the lazy brown dog'

class OrthogonalDecomposition:
    def __init__(self, *, name, ntokens, nterms, dim=768, epsilon=0):
        self.name = name
        self.ntokens = ntokens
        self.dim = dim
        self.max_terms = nterms
        self.epsilon = epsilon
        self.basis_directions = defaultdict(list)
        self.basis_direction_names = defaultdict(list)
        self.blocks = defaultdict(list)
        self.colors = {}
        self.term_names = []

    def add_term(self, name, term):
        #print(f'adding term {name}')
        assert len(term.shape) == 2
        term = term.clone()
        color = nice_color(len(self.term_names), self.max_terms // 2)
        self.colors[name] = color
        self.term_names.append(name)
        for i in range(self.ntokens):
            vec = term[i].clone()
            assert len(vec.shape) == 1

            if len(self.basis_directions[i]) == 0:
                norm = torch.linalg.norm(vec)
                self.basis_directions[i].append(F.normalize(vec, dim=-1))
                self.blocks[i].append((name, name, norm))
                self.basis_direction_names[i].append(name)
            else:
                norm = torch.linalg.norm(vec)
                for j, basis_direction in enumerate(self.basis_directions[i]):
                    proj = torch.dot(basis_direction, vec)
                    vec = vec - proj * basis_direction
                    self.blocks[i].append((name, self.basis_direction_names[i][j], proj))
                if 1: # torch.norm(term[i]) >= self.epsilon:
                    norm = torch.linalg.norm(vec)
                    self.basis_directions[i].append(F.normalize(vec, dim=-1))
                    self.blocks[i].append((name, name, norm))
                    self.basis_direction_names[i].append(name)

    def plot(self, tokens):
        anims = []
        for i in range(self.ntokens):
            fig, ax = plt.subplots(figsize=(5, 5))
            if i < self.ntokens - 1:
                ax.set_title(f'RS usage, token {i} ({tokens[i]} -> {tokens[i+1]}))')
            else:
                ax.set_title(f'RS usage, token {i} ({tokens[i]}))')

            ax.set_xticks(range(len(self.basis_direction_names[i])), self.basis_direction_names[i], rotation=90)
            ax.hlines(0, 0, len(self.basis_direction_names[i]), color='black', zorder=2)
            self.ax = ax
            self.tot_pos = np.zeros(len(self.basis_direction_names[i]))
            self.tot_neg = np.zeros(len(self.basis_direction_names[i]))

            print(f'plotting token {i}')
            nterms = len(self.term_names)
            max_fps = 5
            nouter_frames = (nterms) * (nterms + 2) + 1 # 1 chunk per term, plus 2 chunks of nothing
            self.inner_frames_seen = defaultdict(lambda: False)
            anim = FuncAnimation(fig, partial(self.animate_frame2, i=i), 
                                 frames=nouter_frames, interval=50, repeat=False)
            anims.append(anim)
            anim.save(f'{self.name}-{i}.mp4')
            assert self.last_block == len(self.blocks[i]) - 1
        return anims

    def animate_frame2(self, frame_num, *, i):
        nterms = len(self.term_names)
        if 0: #frame_num % 1 == 0:
            nterms = len(self.term_names)
            print(f'outer frame {frame_num}/{(nterms + 1) ** 2}')
        # make it so that each term occupies the same number of frames
        outer_frame = len(self.term_names) # 1 second of nothing
        for inner_frame, block in enumerate(self.blocks[i]):
            if outer_frame >= frame_num:
                self.animate_frame(inner_frame, i=i)
                return
            idx = self.term_names.index(block[0])
            num = len([x for x in self.blocks[i] if x[0] == block[0]])
            outer_frame += nterms / num
        self.animate_frame(len(self.blocks[i]), i=i)

    def animate_frame(self, frame_num, *, i):
        if 0: #frame_num % 1 == 0:
            print(f'inner frame {frame_num}/{len(self.blocks[i])} for token {i}')
        # stacked bar plot with positive and negative values  
        if self.inner_frames_seen[frame_num]:
            return
        self.inner_frames_seen[frame_num] = True          
        for iblock, block in enumerate(self.blocks[i]):
            if iblock >= frame_num:
                break
            if iblock < frame_num - 1:
                continue
            termname, dirname, score = block
            self.ax.legend([mpatches.Patch(color=self.colors[termname])], [termname])
            idx = self.basis_direction_names[i].index(dirname)
            color = self.colors[termname]
            if score > 0:
                self.ax.bar(idx, score, bottom=self.tot_pos[idx], color=color)
                self.tot_pos[idx] += score
            elif score < 0:
                self.tot_neg[idx] += score
                self.ax.bar(idx, -score, bottom=self.tot_neg[idx], color=color)
        self.last_block = iblock                


def compute_orthogonal_decomposition(*, model, text, make_plot=False):
    tokens = get_tokens(model=model, text=text)
    str_tokens = get_str_tokens(model=model, text=text)
    logits, cache = model.run_with_cache(tokens)
    od = OrthogonalDecomposition(name=model.cfg.model_name, ntokens=len(tokens[0]), 
                                    nterms=model.cfg.n_layers * 2 + 2)
    od.add_term('embed', cache['hook_embed'].squeeze(0))
    od.add_term('pos', cache['hook_pos_embed'].squeeze(0))
    for layer in range(model.cfg.n_layers):
        attn_term = cache[f'blocks.{layer}.hook_attn_out'].squeeze(0)
        od.add_term(f'attn.{layer}', attn_term)
        mlp_term = cache[f'blocks.{layer}.hook_mlp_out'].squeeze(0)
        od.add_term(f'mlp.{layer}', mlp_term)
    if make_plot:
        od.plot(tokens=str_tokens)
    return od


def compute_od_statistics(*, model, texts):
    all_scores = defaultdict(list)
    for text in tqdm(texts):
        od = compute_orthogonal_decomposition(model, text)
        for i in range(od.ntokens):            
            for block in od.blocks[i]:
                term, dir, score = block
                score = score.item()
                all_scores[(term, dir)].append(score)

    all_terms = ['embed', 'pos']
    for layer in range(model.cfg.n_layers):
        all_terms.append(f'attn.{layer}')
        all_terms.append(f'mlp.{layer}')

    fix, axes = plt.subplots(len(all_terms), len(all_terms), figsize=(100, 100))
    for iterm, term in enumerate(tqdm(all_terms)):
        for idir, dir in enumerate(all_terms):
            ax = axes[iterm, idir]
            ax.set_title(f'term={term} dir={dir}')
            scores = np.array(all_scores[(term, dir)])
            scores = scores[~detect_outliers(scores)]
            if np.mean(scores) >= 0:
                retval = ax.hist(scores, bins=100)
                max_height = max(retval[0])
                ax.vlines(0, 0, max_height, color='red')
            else:
                retval = ax.hist(scores, bins=100, color='red')
                max_height = max(retval[0])
                ax.vlines(0, 0, max_height, color='blue')

def compute_orthogonal_decomposition2(*, model, text):
    """Compute the orthogonal decomposition of a model's attention heads.
    This is a reimplementation to check for bugs in the other implementation."""
    tokens = get_tokens(model=model, text=text)
    str_tokens = get_str_tokens(model=model, text=text)
    logits, cache = model.run_with_cache(tokens)
    terms = []
    terms.append(cache['hook_embed'].squeeze(0))
    terms.append(cache['hook_pos_embed'].squeeze(0))
    for layer in range(model.cfg.n_layers):
        attn_term = cache[f'blocks.{layer}.hook_attn_out'].squeeze(0)
        terms.append(attn_term)
        mlp_term = cache[f'blocks.{layer}.hook_mlp_out'].squeeze(0)
        terms.append(mlp_term)

    for i in range(len(tokens[0])):
        print(f'token {i}: {str_tokens[i]}')
        for j in range(len(terms)):
            print(f'term {j}: {terms[j][i].norm().item()}')
        print()