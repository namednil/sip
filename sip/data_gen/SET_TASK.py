import random

def set_task(s):
  """
	Example: aabbbcbbbb -> abc
	"""
  char_set = set()
  new_s = []
  for i in s:
    if i in char_set:
      continue
    else:
      char_set.add(i)
      new_s.append(i)
  return "".join(new_s)


def my_set_task(min_length, max_length, vocab):
    # Following the description in appendix G of https://arxiv.org/pdf/2206.10139.pdf
    length = random.randint(min_length, max_length)
    num_types = random.randint(1, length)
    will_occur = random.sample(vocab, k=num_types) #select a subset of the tokens
    
    parts = []
    for typ in will_occur[:-1]:
        num_types -= 1
        # select at least one instance for every type, 
        # but we can also fill up the entire length (but leaving space for the other types to contribute at least one)
        count = random.randint(1, length-len(parts)-num_types)
        parts.extend(count * [typ])
    #Last type needs to fill the gap
    parts.extend((length-len(parts)) * [will_occur[-1]])
    assert len(parts) == length, f"expected {length}, got {len(parts)}"
    assert set(parts) == set(will_occur)
    random.shuffle(parts)
    return parts
        
        
def write_tsv(fname, inps):
    with open(fname, "w") as f:
        for inp in inps:
            f.write(inp)
            f.write("\t")
            f.write(set_task(inp))
            f.write("\n")

random.seed(3293345)

vocab = [chr(x) for x in range(32, 127)]
vocab = vocab + [chr(i) for i in range(592, 687+1)] # add unicode characters for IPA symbols.
vocab = sorted(set(vocab))
#these characters have special meaning in OpenFST, and cannot be written into FAR
vocab.remove("]")
vocab.remove("[")
vocab.remove(chr(92)) # backslash, this messes things up as well!

data = set()
num_train = 200_000
for _ in range(2*num_train):
    inp = "".join(my_set_task(1,35,vocab))
    data.add(inp)
    
data = sorted(data)
random.shuffle(data)
train_data = data[:num_train]
test_data = data[num_train:num_train+1000]

write_tsv(f"data/pretrain/set_task_{int(num_train/1000)}k_train.tsv", train_data)
write_tsv(f"data/pretrain/set_task_{int(num_train/1000)}k_dev.tsv", test_data)


