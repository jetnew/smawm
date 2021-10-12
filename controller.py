""" Define controller """
import torch
import torch.nn as nn
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid, getcwd
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
import cma
from tqdm import tqdm
import numpy as np
from misc import load_parameters, flatten_parameters, save_checkpoint
from vae import VAE
from mdrnn import MDRNNCell
import gym
from pettingzoo.mpe import simple_adversary_v2
import random


class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)
        self.act = nn.Softmax(dim=1)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.act(self.fc(cat_in))
        
        
class RolloutGenerator(object):
    """ Utility to generate rollouts.
    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.
    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(10, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        self.controller = Controller(LSIZE, RSIZE, 5).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = simple_adversary_v2.env(N=2, max_cycles=time_limit, continuous_actions=False)
        self.device = device

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.
        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.
        :args params: parameters as a single 1D np array
        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        self.env.reset()

        agent_idx = {}
        for i, agent in enumerate(self.env.agents):
            agent_idx[agent] = i

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]
        
        actions = []
        cumulative = 0
        i = 0
        for agent in self.env.agent_iter():
            observation, reward, done, info = self.env.last()
            idx = agent_idx[agent]
            
            # If the agent is not the adversary, apply the trained policy.
            if idx != 0:
                obs = torch.from_numpy(observation).unsqueeze(0).to(self.device)
                _, latent_mu, _ = self.vae(obs)
                action = self.controller(latent_mu, hidden[0])
                action = torch.argmax(action).item()
            
            # Take random actions as the adversary.
            else:
                action = random.randint(0,4) if not done else None
            
            self.env.step(action)
            actions.append(torch.eye(5, device=self.device)[action])
            
            # At the end of all agents, at the last agent's turn, update the MDRNN's hidden state.
            if idx == self.env.max_num_agents - 1:
                actions = torch.cat(actions, 0).unsqueeze(0)
                _, _, _, _, _, hidden = self.mdrnn(actions, latent_mu, hidden)
                actions = []

            if render:
                self.env.render(mode='human')
            if idx != 0:
                cumulative += reward
            if done:
                return - cumulative
            i += 1


################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index):
    """ Thread routine.
    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.
    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.
    As soon as e_queue  is non empty, the thread terminate.
    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).
    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    if torch.cuda.is_available():
        gpu = p_index % torch.cuda.device_count()
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    # redirect streams
    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

    with torch.no_grad():
        r_gen = RolloutGenerator(logdir, device, time_limit)

        while e_queue.empty():
            if p_queue.empty():
                sleep(.1)
            else:
                s_id, params = p_queue.get()
                r_queue.put((s_id, r_gen.rollout(params)))


ASIZE, RSIZE, LSIZE = 15, 30, 15
display = False

# multiprocessing variables
n_samples = 4
pop_size = 4
num_workers = min(32, n_samples * pop_size)
time_limit = 100

# create tmp dir if non existent and clean it if existent
logdir = 'exp_dir'
tmp_dir = join(getcwd(), logdir, 'tmp')


if __name__ == "__main__":
    if not exists(tmp_dir):
        mkdir(tmp_dir)
    else:
        for fname in listdir(tmp_dir):
            unlink(join(tmp_dir, fname))
            
    # create ctrl dir if non exitent
    ctrl_dir = join(getcwd(), logdir, 'ctrl')
    if not exists(ctrl_dir):
        mkdir(ctrl_dir)
        
                    
    ################################################################################
    #                Define queues and start workers                               #
    ################################################################################

    p_queue = Queue()
    r_queue = Queue()
    e_queue = Queue()

    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()
        
        
    ################################################################################
    #                           Evaluation                                         #
    ################################################################################
    def evaluate(solutions, results, rollouts=100):
        """ Give current controller evaluation.
        Evaluation is minus the cumulated reward averaged over rollout runs.
        :args solutions: CMA set of solutions
        :args results: corresponding results
        :args rollouts: number of rollouts
        :returns: minus averaged cumulated reward
        """
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        for s_id in range(rollouts):
            p_queue.put((s_id, best_guess))

        for _ in tqdm(range(rollouts)):
            while r_queue.empty():
                sleep(.1)
            restimates.append(r_queue.get()[1])

        return best_guess, np.mean(restimates), np.std(restimates)


        
    ################################################################################
    #                           Launch CMA                                         #
    ################################################################################
    controller = Controller(LSIZE, RSIZE, 5)  # dummy instance

    # define current best and load parameters
    cur_best = None
    ctrl_file = join(ctrl_dir, 'best.tar')
    print("Attempting to load previous best...")
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = - state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Previous best was {}...".format(-cur_best))

    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                  {'popsize': pop_size})

    epoch = 0
    log_step = 3
    target_return = 300
    while not es.stop():
        if cur_best is not None and - cur_best > target_return:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size  # result list
        solutions = es.ask()

        # push parameters to queue
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        # retrieve results
        if display:
            pbar = tqdm(total=pop_size * n_samples)
        for _ in range(pop_size * n_samples):
            while r_queue.empty():
                sleep(.1)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples
            if display:
                pbar.update(1)
        if display:
            pbar.close()

        es.tell(solutions, r_list)
        es.disp()

        # evaluation and saving
        if epoch % log_step == log_step - 1:
            best_params, best, std_best = evaluate(solutions, r_list)
            print("Current evaluation: {}".format(best))
            if not cur_best or cur_best > best:
                cur_best = best
                print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                load_parameters(best_params, controller)
                torch.save(
                    {'epoch': epoch,
                     'reward': - cur_best,
                     'state_dict': controller.state_dict()},
                    join(ctrl_dir, 'best.tar'))
            if - best > target_return:
                print("Terminating controller training with value {}...".format(best))
                break


        epoch += 1

    es.result_pretty()
    e_queue.put('EOP')
