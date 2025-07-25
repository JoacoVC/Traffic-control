
import os
import pickle
from sumo_rl import SumoEnvironment
from abc import ABC, abstractmethod
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda



class LearningAgent(ABC):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        
        
        self.config = config
        self.env = env
        self.agent = None
        self.name = name

    @abstractmethod
    def _init_agent(self):
        
        pass

    def get_name(self) -> str:
        
        return self.name

    @abstractmethod
    def run(self, learn: bool, out_path: str) -> str:
       
        pass

    def save(self, path: str) -> None:
        
        pass

    def load(self, path: str, env: SumoEnvironment) -> None:
        
        pass

class FixedCycle(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
        
        super().__init__(config, env, name)

    def _init_agent(self):
        self.agent = None

    def run(self, learn: bool, out_path: str) -> str:
        

        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):
            done = False
            self.env.reset()
            while not done:
                done = self._step()
            self.env.save_csv(out_file, curr_run)

        self.env.close()

        return out_path

    def save(self, path: str) -> None:
        
        pass

    def load(self, path: str, env: SumoEnvironment) -> None:
        
        pass

    def _step(self) -> bool:
        
        for _ in range(self.env.delta_time):
            self.env._sumo_step()
        self.env._compute_observations()
        self.env._compute_rewards()
        self.env._compute_info()
        return self.env._compute_dones()['__all__']

class QLearningAgent(LearningAgent):

    def __init__(self, config: dict, env: SumoEnvironment, name: str):
       
        super().__init__(config, env, name)

    def _init_agent(self):
        

        self.agent = QLAgent(
            starting_state=self.env.encode(self.env.reset()[0], self.env.ts_ids[0]),
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=self.config['Alpha'],
            gamma=self.config['Gamma'],
            exploration_strategy=EpsilonGreedy(initial_epsilon=self.config['Init_epsilon'],
                                               min_epsilon=self.config['Min_epsilon'],
                                               decay=self.config['Decay']
                                               ))

    def run(self, learn: bool, out_path: str) -> str:
        
        if self.agent is None:
            self._init_agent()

        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):

            done = False
            while not done:
                state, reward, _, done, _ = self.env.step(self.agent.act())
                if learn:
                    self.agent.learn(self.env.encode(state, self.env.ts_ids[0]), reward)

            self.env.save_csv(out_file, curr_run)
            self.env.reset()
        self.env.close()

        return out_path

    def save(self, path: str) -> None:
        
        data = {
            'alpha': self.agent.alpha,
            'gamma': self.agent.gamma,
            'exploration_strategy': self.agent.exploration,
            'q_table': self.agent.q_table}

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str, env: SumoEnvironment) -> None:
        
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.env = env

        self.agent = QLAgent(
            starting_state=self.env.encode(self.env.reset()[0], self.env.ts_ids[0]),
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=data['alpha'],
            gamma=data['gamma'],
            exploration_strategy=data['exploration_strategy']
        )
        self.agent.q_table = data['q_table']

class SarsaAgent(LearningAgent):
    
    def __init__(self, config: dict, env: SumoEnvironment, name: str):
       
        super().__init__(config, env, name)

    def _init_agent(self):
        
        self.agent = TrueOnlineSarsaLambda(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=self.config['Alpha'],
            gamma=self.config['Gamma'],
            epsilon=self.config['Epsilon'],
            fourier_order=self.config['FourierOrder'],
            lamb=self.config['Lambda']
        )

    def run(self, learn: bool, out_path: str) -> str:
        
        if self.agent is None:
            self._init_agent()

        out_path = os.path.join(out_path, self.name)
        out_file = os.path.join(out_path, self.name)

        for curr_run in range(self.config['Runs']):
            obs, _ = self.env.reset()
            terminated, truncated = False, False
            
            while not (terminated or truncated):
                action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action=action)

                self.agent.learn(state=obs, action=action, reward=reward, next_state=next_obs, done=terminated)
                obs = next_obs

            self.env.save_csv(out_file, curr_run)
            self.env.reset()
        self.env.close()

        return out_path
    

    def save(self, path: str) -> None:
        
        data = {
            'alpha': self.agent.alpha,
            'gamma': self.agent.gamma,
            'epsilon': self.agent.epsilon,
            'lamb': self.agent.lamb,
            'fourier_order': self.agent.basis.order,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str, env: SumoEnvironment) -> None:
      
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.env = env

        self.agent = TrueOnlineSarsaLambda(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=data['alpha'],
            gamma=data['gamma'],
            epsilon=data['epsilon'],
            fourier_order=data['fourier_order'],
            lamb=data['lamb']
        )
