import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

############################################
#from stable_baselines3.common.policies_MOD import ActorCriticPolicy
#from stable_baselines3.common.buffers_MOD import DictRolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor

##added afterwards
import gym
#import time
from stable_baselines3.common.utils import safe_mean
import cv2
#from stable_baselines3.common import utils
import time
from functools import reduce 
from stable_baselines3.common.utils import (
    get_device,
    
)
import cv2

class GEN_POLICY():
    

    def __init__(
        self,HL,LL,env,num_envs,config,device=0
        
    ):

        super(GEN_POLICY, self).__init__()

       
        
        self.low_level_policy = LL
        self.high_level_policy = HL
        
        self.env = env

        self.device = get_device(device)
        self.num_envs = num_envs

        self.low_level_train = True
        self.low_n_steps = 2048

        #marked color is withing the cabinet_colors array
        self.cabinet_colors = config.get("cabinet_colors", [])
        #self.cabinet_marked = config.get("cabinet_marked",[])
        self.unknown_color = np.array(config.get("unknown_color", []))[:,np.newaxis]

        self.fr_point = config.get("fr_point", 159)
        self.fr_point_sub = np.array(config.get("fr_point_substitution", []))[:,np.newaxis]

        self.cracker_colors = config.get("cracker_colors",[])
        self.category_found = np.array(config.get("category_found_color",[]))[:,np.newaxis]

        self.corrected_discounting = config.get("corrected_discounting",False)
        
        self.explo_steps = config.get("exploration_policy_steps",False)


    
    def easy_log_ll(self,iteration):
        fps = int(self.low_level_policy.num_timesteps  / (time.time() - self.start_time))
        self.low_level_policy.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.low_level_policy.ep_info_buffer) > 0 and len(self.low_level_policy.ep_info_buffer[0]) > 0:
            self.low_level_policy.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.low_level_policy.ep_info_buffer]))
            self.low_level_policy.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.low_level_policy.ep_info_buffer]))
        self.low_level_policy.logger.record("time/fps", fps)
        self.low_level_policy.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        self.low_level_policy.logger.record("time/total_timesteps", self.low_level_policy.num_timesteps , exclude="tensorboard")
        self.low_level_policy.logger.dump(step=self.low_level_policy.num_timesteps )



    
    def easy_log_hl(self,iteration):
        fps = int(self.high_level_policy.num_timesteps  / (time.time() - self.start_time))
        self.high_level_policy.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.high_level_policy.ep_info_buffer) > 0 and len(self.high_level_policy.ep_info_buffer[0]) > 0:
            self.high_level_policy.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.high_level_policy.ep_info_buffer]))
            self.high_level_policy.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.high_level_policy.ep_info_buffer]))
        self.high_level_policy.logger.record("time/fps", fps)
        self.high_level_policy.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        self.high_level_policy.logger.record("time/total_timesteps", self.high_level_policy.num_timesteps , exclude="tensorboard")
        self.high_level_policy.logger.dump(step=self.high_level_policy.num_timesteps )


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":


        iteration = 0
        
         #_.setup_learn is resonponsible for resetting the env first and set obs to ._last_obs etc. inilizaing the logger etc..
        # now the last env.reset has been done by the high level policy slightly below. Therefore, take it as first init.
        #_, callback_ll = self.low_level_policy._setup_learn(
        #    total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        #)
        
        #self.ll_callback = callback_ll


        #_.setup_learn is resonponsible for resetting the env first and set obs to ._last_obs etc. inilizaing the logger etc..
        total_timesteps, callback_hl = self.high_level_policy._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        self.hl_callback = callback_hl
        
        

       

        
        self.hl_callback.on_training_start(locals(), globals())

        #self.ll_callback.on_training_start(locals(), globals())

        
        self.start_time = time.time()

        while self.high_level_policy.num_timesteps < total_timesteps:
            
            continue_training = self.collect_rollouts(self.env,n_rollout_steps=2048)#, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            
            iteration += 1
            self.easy_log_hl(iteration)
            print("Update HIGH LEVEL POLICY")
            self.high_level_policy.train()

            
            
            

            
        #callback.on_training_end()

        return self
    

    def observation_to_tensorHL(self,obs, device):#,prev_frontier_point):
        """
        hl_obs = {}
        for (key, _obs) in obs.items():
            if key == 'image_global':
                
                for i in range(_obs.shape[0]):
                    p = prev_frontier_point[i]
                    print("whats the point?",p)
                    if p[0] != -99:
                        _obs[i,:,p[0]-5:p[0]+5,p[1]-5:p[1]+5] = self.unknown_color[:,np.newaxis]

                    cv2.imshow("coarse_2_{}".format(i), _obs[i].transpose(1, 2, 0).astype(np.uint8))
                    cv2.waitKey(1)
                hl_obs[key] = th.as_tensor(_obs).to(device)
            elif key == "opening_state":
                hl_obs[key] = th.as_tensor(_obs).to(device)
            elif key == "objects_state":
                hl_obs[key] = th.as_tensor(_obs).to(device)
        """
        return {key: th.as_tensor(_obs).to(device) for (key, _obs) in obs.items() if key in ['image_global',"valid_actions","task_obs_hl"]}

    def observation_to_tensor_ll(self,obs,device,indices):#,nump=False):

        ll_obs = {}
        for (key, _obs) in obs.items():
            
            if key in ['image_global','image']:

                img = _obs[indices]
                
                for singl_img in range(sum(indices)):
                    s_img = img[singl_img]
                    
                    #remove the cabinet colors
                    masks = reduce(np.logical_or, [s_img[2,:,:] == col for col in self.cabinet_colors])
                    s_img[:,masks] = self.unknown_color

                    mask = s_img[0,:,:] == self.fr_point
                    s_img[:,mask] = self.fr_point_sub

                    #last three entries in "task_obs" are the cracker boxes
                    already_found = obs['task_obs'][singl_img][-3::]
                    #print("FOund ?",already_found)
                    cracker_re_coloring = [s_img[0,:,:] == col for i,col in enumerate(self.cracker_colors) if already_found[i] == 0]
                    if len(cracker_re_coloring) != 0:
                        masks = reduce(np.logical_or,cracker_re_coloring) 
                        s_img[:,masks] =  self.category_found                  

                    img[singl_img] = s_img

                    

                    #self.frontier_point = np.array([159, 79, 122])
                    #self.frontier_point_pruned = np.array([159, 161, 122])
                    #if key == "image_global":
                    #    cv2.imshow("coarse_2_{}".format(singl_img), s_img.transpose(1, 2, 0).astype(np.uint8))
                    #    cv2.waitKey(1)

                #if nump:
                #    ll_obs[key] = img
                #else:
                ll_obs[key] = th.as_tensor(img).to(device)

            elif key == "task_obs":
                #if nump:
                #    ll_obs[key] = _obs[indices]
                #else:
                ll_obs[key] = th.as_tensor(_obs[indices]).to(device)

        return ll_obs
        #return {key: th.as_tensor(_obs[indices]).to(device) for (key, _obs) in obs.items() if key in ['image_global','image',"task_obs"]}

    #Overwritten Method since we use it here for auxilliary Task
    def collect_rollouts(
        self,
        env: VecEnv,
        #callback: BaseCallback,
        #rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        #self._last_obs = self.env.reset()
        assert self.high_level_policy._last_obs is not None, "No previous observation was provided"

        self._last_obs = self.high_level_policy._last_obs

        self.low_level_policy._last_obs = self._last_obs
         
        # Switch to eval mode (this affects batch norm / dropout)                                                                                                                                                                               
        self.high_level_policy.policy.set_training_mode(False)


        self.hl_callback.on_rollout_start()
        #self.hl_callback.on_rollout_start()

        #policy_actions = [self.low_level_1(),self.low_level_1_5(),self.low_level_1_5(),self.low_level_1_5(),self.low_level_1_5()]
        #policy_buffers = [add_low_level_1_buffer(),...]

        n_steps = 0
        #self.low_level_policy.rollout_buffer.reset()

        self.high_level_policy.rollout_buffer.reset()
        
        

        #callback.on_rollout_start()
        #total_rew = [0.0,0.0]
        cumulative_rewards = [0.0]*self.num_envs
        #prev_frontier_point = np.array([[-99.0,-99.0]]*self.num_envs)
        while n_steps < n_rollout_steps:

            n_steps += 1
            with th.no_grad():
                #print("step 1")
                #print("PREV:",prev_frontier_point)
                # HL-policy gets only global_image as observation
                obs_tensor = self.observation_to_tensorHL(self._last_obs,self.device)#,prev_frontier_point)#obs_as_tensor(self._last_obs, self.device)
               
                hl_actions, values, log_probs  = self.high_level_policy.policy.forward(obs_tensor)
            

            
            #high level action
            hl_actions = hl_actions.cpu().numpy()
            
            #print("HL-STEPS:",hl_actions)
            #input()
            active_env = set(range(self.num_envs))
            low_level_actions = np.empty((self.num_envs,1),dtype=object)
            if self.corrected_discounting:
                discount_length = np.empty((self.num_envs,1),dtype=object)

            #maintain own self.info because the VecMonitor returns valueble information about the 
            #episode which is used in the ep_rew print information. Now the sub_proc_env_HRL overwrites theese
            #information when it was not the 8th step. Therefore, save own information.
            self.infos = np.array([{}]*self.num_envs)
            #print("HL-actions:",hl_actions)
            #loop all low level policies for at least 8 steps
            for ll_step in range(self.explo_steps):
                #actions_together = []

                
                ll_1_indices = hl_actions == 0
             

                ll_1_indices_length = ll_1_indices.sum()
                if ll_1_indices_length != 0:
                    ll_1_actions = self.low_level_0(self.observation_to_tensor_ll(self._last_obs,self.device,ll_1_indices),ll_1_indices_length)#,prev_frontier_point,ll_1_indices)   
                    low_level_actions[ll_1_indices] = ll_1_actions
                
                #other policies
                ll_2_6_indices = hl_actions != 0
                ll_2_6_indices_length = ll_2_6_indices.sum()
                if ll_2_6_indices_length != 0:
                    #currently doesn't need any observation since it is executed in the env
                    ll_2_6_actions = self.low_level_1_5(hl_actions[ll_2_6_indices],ll_2_6_indices_length)#,prev_frontier_point,ll_2_6_indices)#policy_actions[1](self._last_obs[ll_2_6_indices,...])
                    low_level_actions[ll_2_6_indices] = ll_2_6_actions

                #frontier point should be always in action_space
                
               
                #print("actions:",low_level_actions)
                new_obs, rewards, dones, infos = self.env.step(low_level_actions.squeeze(1), indices=active_env)
                
                
                #prev_frontier_point = np.array([infos[i]['frontier_point'] for i in range(self.num_envs)])
                #prev_frontier_point[dones==True] = np.array([-99,-99],dtype=int)
                if ll_step == 0:
                    cumulative_rewards = rewards
                    self.infos = np.array(infos)
                    if self.corrected_discounting:
                        discount_length = np.array([infos[i]['discount_length'] for i in range(self.num_envs)])#infos['discount_length']
                else:
                    cumulative_rewards[list(active_env)] += rewards[list(active_env)] #* self.gamma**count#self._last_episode_starts * self.gamma**count #use high level gamma since this is used in the training
                    self.infos[list(active_env)] = np.array(infos)[list(active_env)]
                    if self.corrected_discounting:
                        discount_length[list(active_env)] += [infos[i]['discount_length'] for i in list(active_env)]
                    
                
                active_env -= set(list(np.where(ll_2_6_indices)[0]))
                active_env -= set(list(np.where(dones==True)[0]))
                #print("active after",active_env)
                #total_rew += cumulative_rewards
                """
                if (dones==True).sum() > 0:
                    print("Done")
                    if not("episode" in self.infos[0] or "episode" in self.infos[1]):

                        print("one is done")
                        print("info",infos)
                        print("rew:",cumulative_rewards)
                        #print("total:",total_rew)
                        #total_rew = [0.0,0.0]

                        input()
                """
                #print("cum-reward",cumulative_rewards)

                #train low level
                #aux_angle_gt = infos[ll_1_indices]['true_aux'][1]
                #self.add_low_level_0_buffer(new_obs[ll_1_indices],rewards[ll_1_indices],actions,last_episode_start,values,log_probs,aux_angle,aux_angle_gt,dones[ll_1_indices],ll_1_indices_length)


                self.low_level_policy._last_obs = new_obs
                self._last_obs = new_obs
                if len(active_env) == 0:
                    break

                
           
            self.hl_callback.update_locals(locals())
            #needs to stay there in order to save the model
            self.hl_callback.on_step()
            #if self.hl_callback.on_step() is False:
            #    pass
                #return False
            
            #print("add buffer:\n")
            #print(f"hl-actions: {hl_actions} and discount_length: {discount_length} and rewards {cumulative_rewards}")
            #input()
            if self.corrected_discounting:
                self.add_low_level_1_5_buffer(new_obs,cumulative_rewards,hl_actions,values,log_probs,dones,list(self.infos),discount_length)
            else:
                self.add_low_level_1_5_buffer(new_obs,cumulative_rewards,hl_actions,values,log_probs,dones,list(self.infos))
            #print("reset")
            

            #input()
            #cumulative_rewards was one -2.5025 ,how did that happen ?
            
            
            

        
        
        
        #compute the advantage estimate after enough rollouts have been collected and the high level policy gets trained
        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = self.observation_to_tensorHL(self._last_obs,self.device)#,prev_frontier_point)
            #added _ here Fabi
            
            _, values, _ = self.high_level_policy.policy.forward(obs_tensor)

        self.high_level_policy.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)



        self.hl_callback.on_rollout_end()
        
        return True

    def add_low_level_0_buffer(self,new_obs,rewards,actions,values,log_probs,aux_angle,aux_angle_gt,dones,length,infos):
        self.low_level_policy._update_info_buffer(infos)
        self.low_level_policy.rollout_buffer.add2(self.low_level_policy._last_obs, actions, rewards, self.low_level_policy._last_episode_starts, values, log_probs,aux_angle,aux_angle_gt)
        self.low_level_policy._last_obs = new_obs
        self.low_level_policy._last_episode_starts = dones
        self.low_level_policy.num_timesteps += length

        self.check_ll_0_train()


    def add_low_level_1_5_buffer(self,new_obs,rewards,actions,values,log_probs,dones,infos, discount_length=None):

        self.high_level_policy.num_timesteps += self.num_envs
        self.high_level_policy._update_info_buffer(infos)

        #if isinstance(self.action_space, gym.spaces.Discrete):
        # Reshape in case of discrete action
        actions = actions.reshape(-1, 1)
        if self.corrected_discounting:
            self.high_level_policy.rollout_buffer.add(self.high_level_policy._last_obs, actions, rewards, self.high_level_policy._last_episode_starts, values, log_probs,discount_length)
        else:
            self.high_level_policy.rollout_buffer.add(self.high_level_policy._last_obs, actions, rewards, self.high_level_policy._last_episode_starts, values, log_probs)
        self.high_level_policy._last_obs = new_obs
        self.high_level_policy._last_episode_starts = dones
        

    def check_ll_0_train(self):
            
        
        self.easy_log_ll(iteration)  
            
        if (self.low_level_policy.num_timesteps // self.env.num_envs) % self.low_n_steps == 0:
            print("update")
            with th.no_grad():
                obs_tensor = self.low_level_policy._last_obs
                obs_tensor= obs_as_tensor(obs_tensor,self.device)
                _, values, _,_,_ = self.low_level_policy.policy.forward(obs_tensor)
                self.low_level_policy.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
            
            #self.log_low_level()
            self.low_level_policy.train()
            self.low_level_policy.rollout_buffer.reset()

    def low_level_0(self,obs,length):#,nump=False):#,prev_frontier_point,ll_1_indices):
       
        with th.no_grad():
            #potentially remove obs_tensor entries for high level 1 actions to save some memory and computational costs (probably neglectable)
            #if nump:
            #    actions, _,aux_angle = self.low_level_policy.predict(obs)
            #else:
            actions, _, _,aux_angle = self.low_level_policy.policy.forward(obs)
            actions = actions.cpu().numpy()
            aux_angle = aux_angle.cpu().numpy()
        aux_actions = [[{"action":actions[i],"aux_angle":aux_angle[i],"hl_action":0}] for i in range(length)]#,"prev_frontier":prev_frontier_point[ll_1_indices]
        #self.low_level.num_timesteps += 1
        return aux_actions

    def low_level_1_5(self,hl_action,length):#,prev_frontier_point,ll_2_6_indices):
        #16 reflects the aux bin size
        #print("indices:",prev_frontier_point,ll_2_6_indices)
        #print(prev_frontier_point[ll_2_6_indices])
        return np.array([[{"action":[0.0,0.0],"aux_angle":[0.0]*12,"hl_action":hl_action[i]}] for i in range(length)])#,"prev_frontier":prev_frontier_point[ll_2_6_indices]
 

    def predict(self,obs,hl_action,man_ac=None):
        
        if hl_action[0] == 0:
            
            img = obs['image_global']
            already_found = obs['task_obs'][-3::]
            cracker_re_coloring = [img[0,:,:] == col for i,col in enumerate(self.cracker_colors) if already_found[i] == 0]
            if len(cracker_re_coloring) != 0:
                masks = reduce(np.logical_or,cracker_re_coloring) 
                img[:,masks] =  self.category_found   
            
            masks = reduce(np.logical_or, [img[2,:,:] == col for col in self.cabinet_colors])
            img[:,masks] = self.unknown_color
            mask = img[0,:,:] == self.fr_point
            img[:,mask] = self.fr_point_sub     
            obs['image_global'] = img

            img = obs['image']
            cracker_re_coloring = [img[0,:,:] == col for i,col in enumerate(self.cracker_colors) if already_found[i] == 0]
            if len(cracker_re_coloring) != 0:
                masks = reduce(np.logical_or,cracker_re_coloring) 
                img[:,masks] =  self.category_found  

            masks = reduce(np.logical_or, [img[2,:,:] == col for col in self.cabinet_colors])
            img[:,masks] = self.unknown_color

            mask = img[0,:,:] == self.fr_point
            #print("fr-color : ",self.fr_point,mask.sum())
            img[:,mask] = self.fr_point_sub     
            obs['image'] = img


            
            
            #print("FOund ?",already_found)
            
                  

            actions, _ , aux_angle = self.low_level_policy.predict(obs)


            #cv2.imshow("coarse_2", obs['image'].transpose(1, 2, 0).astype(np.uint8))
            #cv2.waitKey(1)

            #cv2.imshow("global_2", obs['image_global'].transpose(1, 2, 0).astype(np.uint8))
            #cv2.waitKey(1)
            
            #actions = actions.cpu().numpy()
            #aux_angle = aux_angle.cpu().numpy()
            aux_actions = {"action":actions,"aux_angle":aux_angle[0],"hl_action":0}
            return aux_actions
        else:
            return {"action":np.array([0.0,0.0]),"aux_angle":np.array([0.0]*12),"hl_action":hl_action[0]}#self.low_level_1_5(hl_action,1).squeeze(1).squeeze(0)