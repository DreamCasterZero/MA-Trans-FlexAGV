import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.states        = []
        self.padding_masks = []
        self.action_masks  = []
        self.actions       = []
        self.log_probs     = []
        self.values        = []
        self.rewards       = []
        self.dones         = []

    def push(self, state, padding_mask, action_mask, action, log_prob, value, reward, done):
        self.states.append(state)
        self.padding_masks.append(padding_mask)
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, mini_batch_size):
        n       = len(self.states)
        indices = np.random.permutation(n)
        batches = [indices[i:i + mini_batch_size] for i in range(0, n, mini_batch_size)]

        to_np = lambda lst: np.array([x.cpu().detach().numpy() for x in lst])

        return (
            to_np(self.states),
            to_np(self.padding_masks),
            to_np(self.action_masks),
            np.array(self.actions),
            to_np(self.log_probs),
            to_np(self.values),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones,   dtype=np.float32),
            batches,
        )

    def clear(self):
        self.states.clear()
        self.padding_masks.clear()
        self.action_masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)
