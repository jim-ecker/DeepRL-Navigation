# DeepRL-Navigation

![Banana World Gif](https://github.com/jim-ecker/DeepRL-Navigation/raw/master/images/bananaworld.gif)
## Banana World
Agent must collect bananas in a large, square room.
#### Task Type
Episodic
#### Rewards
__+1__ for each yellow banana

__-1__ for each blue banana
#### Goal
 
Navigate through the room, collecting as many yellow bananas as possible while avoiding the blue bananas

__This environment is considered "solved" when the agent recieves a scoreof 13 points, averaged over 100 episodes__

#### State Space
* ##### State Size
  37 Dimensions
* ##### Space Type
  Continuous
* ##### Representation
  Agent Velocity

  Ray traces to objects around agent's forward direction
#### Action Space
* ##### Space Size
  4 Dimensions
* ##### Space Type
  Discrete space
* ##### Repesentation
  0 | 1 | 2 | 3
  --|---|---|--
  forward | backward | left | right
  
## Installation
In a new virtual environment: 

```bash 
cd {local path to cloned repo}
cd ./python && python setup.py build && python setup.py install && python setup.py clean && cd -
```
## Running the agent
```bash
cd {local path to cloned repo}
python navigation.py {options}
```
* ##### Available Options
  Option | Description
  -------|------------
  --novis | Run without visualization
  --env-dir | Set directory where environment is implemented
  --env-file | Set file for environment
  --n-episodes | Set number of episodes to run agent
  --seed | Set random seed for reproducibility
  --cpu | Use CPU instead of GPU
  --help | Show help message
  ~~--prioritized~~ | ~~**Not Yet Implemented** Use Prioritized Replay Memory~~
  ~~--double~~ | ~~**Not Yet Implemented** Use double deep q network~~
  ~~--dueling~~ | ~~**Not Yet Implemented** Use dueling deep q networks~~
