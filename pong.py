import numpy as np
import gym
import _pickle as pickle

# controls
render = False
resume = True

# hyperparameters
D = 80 * 80
H = 200
gamma = 0.99
learning_rate = 1e-3
decay_rate = 0.99
w1_filename = 'w1st.p'
w2_filename = 'w2st.p'

# model initialization
if resume:
    w1 = pickle.load(open(w1_filename, 'rb'))
    w2 = pickle.load(open(w2_filename, 'rb'))
else:
    w1 = np.random.randn(D, H) / np.sqrt(D) # Xavier initialization
    w2 = np.random.randn(H, 1) / np.sqrt(H)
    
# gradient update initialization
w1_cache = np.zeros_like(w1)
w2_cache = np.zeros_like(w2)

def preprocess(I):
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

prev_frame = np.zeros(D)
def frame_difference(frame):
    global prev_frame
    diff = frame - prev_frame
    prev_frame = frame
    return diff

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(x):
    h = np.maximum(0, np.dot(x, w1))
    logp = np.dot(h, w2)
    p = sigmoid(logp)
    return p, h

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    discounted_rt = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: discounted_rt = 0 # reset at the end of the game when reward -1 or +1
        discounted_rt *= gamma
        discounted_rt += r[t]
        discounted_r[t] = discounted_rt
    return discounted_r

def backward(h, dlogp):
    dw2 = np.dot(h.T, dlogp)
    dh = np.dot(dlogp, w2.T)
    dh[h <= 0] = 0
    dw1 = np.dot(episode_x.T, dh)
    return dw1, dw2

env = gym.make('Pong-v0')
observation = env.reset()

xs, hs, dlogps, rs = [], [], [], []
episode_number = 0
episode_reward = 0
while True:
    if render: env.render()
    
    # model input
    frame = preprocess(observation)
    x = frame_difference(frame)
    xs.append(x)
    
    # sample action
    p, h = forward(x)
    hs.append(h)
    y = 1 if np.random.uniform() < p else 0
    
    # calculate derivative of logp
    dlogp = y - p
    dlogps.append(dlogp)
    
    # make action
    action = 2 if y == 1 else 3 # 1 wait, 2 up, 3 down
    observation, reward, done, info = env.step(action)
    
    # update reward
    episode_reward += reward
    rs.append(reward)
    
    if done:
        episode_number += 1
        
        # episode params
        episode_x = np.vstack(xs)
        episode_h = np.vstack(hs)
        episode_dlogp = np.vstack(dlogps)
        episode_r = np.vstack(rs)
    
        # discount rewards
        discounted_episode_r = discount_rewards(episode_r)
        discounted_episode_r -= np.mean(discounted_episode_r)
        discounted_episode_r /= np.std(discounted_episode_r)
        
        # modulate logprob gradients with rewards
        episode_dlogp *= discounted_episode_r
        
        # calculate gradients
        dw1, dw2 = backward(episode_h, episode_dlogp)
        
        # rmsprop update
        w1_cache = decay_rate * w1_cache + (1 - decay_rate) * dw1**2
        w1 += learning_rate * dw1 / (np.sqrt(w1_cache) + 1e-5)
        w2_cache = decay_rate * w2_cache + (1 - decay_rate) * dw2**2
        w2 += learning_rate * dw2 / (np.sqrt(w2_cache) + 1e-5)
        
        # episode results
        print('episode %d, reward %d' % (episode_number, episode_reward))
        if episode_number % 20: 
            pickle.dump(w1, open(w1_filename, 'wb'))
            pickle.dump(w2, open(w2_filename, 'wb'))
        
        # reset episode params
        xs, hs, dlogps, rs = [], [], [], []
        episode_reward = 0
        prev_frame = np.zeros(D)
        observation = env.reset()