import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
#from src.highlevel_policy.base_vec_env import VecEnv,CloudpickleWrapper,VecEnvIndices,VecEnvObs,VecEnvStepReturn

def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                #print("STEP in Indiv env",data)
                observation, reward, done, info = env.step(data)
                #print("any response ?")
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation = env.reset()
                #print("sending")
                remote.send((observation, reward, done, info))
                #print("send obs rew, done back")
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None, num_discrete_actions: int = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        #action_space  = gym.spaces.Dict({'action': gym.spaces.Box(shape=(2,), low=-1.0, high=1.0, dtype=np.float32),'aux': gym.spaces.Box(shape=(2,), low=-np.inf, high=np.inf, dtype=np.float32)})
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.succ_rate = [[] for _ in range(len(env_fns))]#np.zeros(len(env_fns))
        self.collision_rate = [[] for _ in range(len(env_fns))]
        self.wrong_command = [[] for _ in range(len(env_fns))]
        self.distance_planner = [[] for _ in range(len(env_fns))] 
        self.distance_planner_sum = [[] for _ in range(len(env_fns))]
        self.distance_agent_general = [[] for _ in range(len(env_fns))]
        self.num_discrete_actions = num_discrete_actions#action_space.n
        self.action_rel_freq = [[0.0]*self.num_discrete_actions for _ in range(len(env_fns))]#9actions for the agent
        self.numb_envs = len(env_fns)

    def step(self, actions: np.ndarray,indices: VecEnvIndices = None) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions,indices)
        return self.step_wait(indices)

    def step_async(self, actions: np.ndarray,indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        #print("Iam about to send following actions:",actions)
        #print(f"active env: {indices}")
        for remote, action_ind in zip(target_remotes, indices):
            #print("send action:",actions[action_ind])
            #print(f"action active env {action_ind} and action taken: {actions[action_ind]['hl_action']}")
            self.action_rel_freq[action_ind][actions[action_ind]['hl_action']] += 1
            remote.send(("step", actions[action_ind]))
        #print(f"freq {self.action_rel_freq}")


        self.waiting = True

    def step_wait(self,indices: VecEnvIndices=None) -> VecEnvStepReturn:
        target_remotes = self._get_target_remotes(indices)
        results = [remote.recv() for remote in target_remotes]
        self.waiting = False
        #now for instance, if only 3 out of 8 env are active because of the low level policy
        #the dones array would have just three entries.
        obs, rews, dones, infos = zip(*results)
        #print(len(rews)," ind:",indices)
        #for i,d in enumerate(dones):
        for i,indice in enumerate(indices):
            if(dones[i]):
                #keep history of 35 episodes in buffer
        
                if infos[i]['success']:
                    self.succ_rate[indice].append(1) 
                else:
                    self.succ_rate[indice].append(0) 

                self.wrong_command[indice].append(infos[i]['wrong_command'])
                self.collision_rate[indice].append(infos[i]['collision_step'])
                self.distance_planner[indice].append(infos[i]['distance_planner'])
                #self.distance_planner_sum[indice].append(infos[i]['distance_planner_sum'])
                self.distance_agent_general[indice].append(infos[i]['distance_agent_general'])
                
                if(len(self.collision_rate[indice]) > 36):
                    self.collision_rate[indice].pop(0)
                    self.wrong_command[indice].pop(0)
                    self.succ_rate[indice].pop(0)
                    self.distance_planner[indice].pop(0)
                    #self.distance_planner_sum[indice].pop(0)
                

        if indices is None or len(indices)==self.num_envs:
            self.obs=obs
            self.rews=rews
            self.dones=dones
            self.infos=infos
        else:

            self.obs = np.array(self.obs)
            self.rews = np.array(self.rews)
            self.dones = np.array(self.dones)
            self.infos = np.array(self.infos)
            """
            for i,indice in enumerate(indices):
                
                #print(type(self.obs)) -> tuple
                #print(type(obs)) -> tuple
                #print(type(obs[i])) -> dict
                #print(type(self.obs[indice])) -> dict
                #need to convert it back and forth
                
                self.obs[indice] = obs[i]
                self.rews[indice] = rews[i]
                self.dones[indice] = dones[i]
                self.infos[indice] = infos[i]
            """
            tmp = list(indices)#np.array(list(indices)).astype(int)#,dtype=np.int16)
            
            self.obs[tmp] = obs
            self.rews[tmp] = rews
            self.dones[tmp] = dones
            self.infos[tmp] = infos
            
            self.obs = tuple(self.obs)
            self.rews = tuple(self.rews)
            self.dones = tuple(self.dones)
            self.infos = tuple(self.infos)     
            

        
        #now the dones and rews are filled with the new values correspondingly of what was returned by the workers.
        return _flatten_obs(self.obs, self.observation_space), np.stack(self.rews), np.stack(self.dones), self.infos

    """
    def step_wait(self) -> VecEnvStepReturn:
        #print("waiting for results")
        results = [remote.recv() for remote in self.remotes]
        #print("got the results")
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        for i,inf in enumerate(infos):
            if(dones[i]):
                #keep history of 35 episodes in buffer
                if(len(self.succ_rate[i]) > 35):
                    self.succ_rate[i].pop(0)

                if inf['success']:
                    self.succ_rate[i].append(1) 
                else:
                    self.succ_rate[i].append(0) 

                #keep history of 35 episodes in buffer
                if(len(self.collision_rate[i]) > 35):
                    self.collision_rate[i].pop(0)

                self.collision_rate[i].append(inf['collision_step']) 
               


        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos
    """
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self,indices: VecEnvIndices=None) -> VecEnvObs:
        target_remotes = self._get_target_remotes(indices)
        #for remote in self.remotes:
        for remote in target_remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        #print("INDICES:",indices)
        return [self.remotes[i] for i in indices]

    def reset_action_dist(self):
        self.action_rel_freq = [[0.0]*12 for _ in range(self.numb_envs)] # reset array after plotting, before its used in ppo.py


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: gym.spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
