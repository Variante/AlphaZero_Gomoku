## uArm + AlphaZero-Gomoku

Attach an uArm to put the checkers to the checkerboard. A quick video demo at [Bilibili](https://www.bilibili.com/video/BV1uD4y1a7Ws/).

Debug:
1. Change camera id: `L91` in `RobotManager.py`
2. Change checker color: `L304-309` in `RobotManager.py`
3. Change checkerboard size / rules: `run()` in`real_play.py`

Cmds:
1. c: calibrate the camera
2. r: restart an episode (TODO: automatically collect checkers)
3. f: flip the checkerboard
4. a: allow cheating when detects errors

First time use:
1. Print the checkerboard at this repo(./checkerboard/6x6board.pdf)
2. Print an AprilTag [num 0 in tag36h11](https://github.com/AprilRobotics/apriltag-imgs/blob/master/tag36h11/tag36_11_00000.png) and put it on top of the gripper
3. Set up the camera so it can see the checkerboard and the moving robot gripper
4. Run calibration (press c in command line)

For every episode:
1. User put the checker and press any key (except c, r, f) to continue 
2. The robot starts a move
3. If you wins the robot will flip the checkerboard by default

---
Here comes the original readme
---
## AlphaZero-Gomoku
This is an implementation of the AlphaZero algorithm for playing the simple board game Gomoku (also called Gobang or Five in a Row) from pure self-play training. The game Gomoku is much simpler than Go or chess, so that we can focus on the training scheme of AlphaZero and obtain a pretty good AI model on a single PC in a few hours. 

References:  
1. AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
2. AlphaGo Zero: Mastering the game of Go without human knowledge

### Update 2018.2.24: supports training with TensorFlow!
### Update 2018.1.17: supports training with PyTorch!

### Example Games Between Trained Models
- Each move with 400 MCTS playouts:  
![playout400](https://raw.githubusercontent.com/junxiaosong/AlphaZero_Gomoku/master/playout400.gif)

### Requirements
To play with the trained AI models, only need:
- Python >= 2.7
- Numpy >= 1.11

To train the AI model from scratch, further need, either:
- Theano >= 0.7 and Lasagne >= 0.1      
or
- PyTorch >= 0.2.0    
or
- TensorFlow

**PS**: if your Theano's version > 0.7, please follow this [issue](https://github.com/aigamedev/scikit-neuralnetwork/issues/235) to install Lasagne,  
otherwise, force pip to downgrade Theano to 0.7 ``pip install --upgrade theano==0.7.0``

If you would like to train the model using other DL frameworks, you only need to rewrite policy_value_net.py.

### Getting Started
To play with provided models, run the following script from the directory:  
```
python human_play.py  
```
You may modify human_play.py to try different provided models or the pure MCTS.

To train the AI model from scratch, with Theano and Lasagne, directly run:   
```
python train.py
```
With PyTorch or TensorFlow, first modify the file [train.py](https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/train.py), i.e., comment the line
```
from policy_value_net import PolicyValueNet  # Theano and Lasagne
```
and uncomment the line 
```
# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
or
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
```
and then execute: ``python train.py``  (To use GPU in PyTorch, set ``use_gpu=True`` and use ``return loss.item(), entropy.item()`` in function train_step in policy_value_net_pytorch.py if your pytorch version is greater than 0.5)

The models (best_policy.model and current_policy.model) will be saved every a few updates (default 50).  

**Note:** the 4 provided models were trained using Theano/Lasagne, to use them with PyTorch, please refer to [issue 5](https://github.com/junxiaosong/AlphaZero_Gomoku/issues/5).

**Tips for training:**
1. It is good to start with a 6 * 6 board and 4 in a row. For this case, we may obtain a reasonably good model within 500~1000 self-play games in about 2 hours.
2. For the case of 8 * 8 board and 5 in a row, it may need 2000~3000 self-play games to get a good model, and it may take about 2 days on a single PC.

### Further reading
My article describing some details about the implementation in Chinese: [https://zhuanlan.zhihu.com/p/32089487](https://zhuanlan.zhihu.com/p/32089487) 
