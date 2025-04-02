import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import cv2
from ultralytics import YOLO
from torchvision import transforms
import os

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        return state, action, next_state, reward, done
    
    def __len__(self):
        return len(self.memory)

class RLPAgent:
    def __init__(self, state_size, action_size, grid_size, grayscale_values, max_steps, detector_model):
        self.state_size = state_size
        self.action_size = action_size
        self.grid_size = grid_size
        self.grayscale_values = grayscale_values
        self.max_steps = max_steps
        self.detector_model = detector_model
        
        # DQN networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(10000)
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.02
        self.batch_size = 64
        self.target_update = 10
        
        self.epsilon = self.epsilon_start
        self.steps_done = 0
        
    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(torch.FloatTensor(state)).argmax().item()
        else:
            return random.randrange(self.action_size)
    
    def update_epsilon(self, episode):
        # Exponential epsilon decay
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-episode * self.epsilon_decay)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Compute Q(s_t, a)
        current_q = self.policy_net(states).gather(1, actions)
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        
        # Compute the expected Q values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class RLPEnvironment:
    def __init__(self, grid_size, grayscale_values, obj_detector):
        """
        Initialize the RLP environment.
        
        Args:
            grid_size: Tuple (m,m) where m is the grid dimension
            grayscale_values: List of grayscale values for the patches
            obj_detector: Object detector model
        """
        self.grid_size = grid_size
        self.m = grid_size[0]  # Grid dimension
        self.grayscale_values = grayscale_values
        self.detector = obj_detector
        
        # Initialize grid state (0 means no patch)
        self.grid = np.zeros(grid_size)
        self.current_pos = (0, 0)
        self.patch_blocks = []
        self.current_image = None
        self.original_confidence = 0
        self.current_confidence = 0
        self.step_count = 0
        
    def reset(self, image, bbox):
        """
        Reset environment with a new image and bounding box.
        
        Args:
            image: Original clean image
            bbox: Bounding box of the target object [x1, y1, x2, y2]
        
        Returns:
            Initial state
        """
        self.grid = np.zeros(self.grid_size)
        self.current_image = image.copy()
        self.bbox = bbox
        self.patch_blocks = []
        
        # Calculate the block size based on the bounding box
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        self.block_size = int(min(w, h) * 0.26 / self.m)
        
        # Set starting position (at 0.2h from the top-left of bbox)
        start_x = bbox[0] + int(0.2 * h)
        start_y = bbox[1] + int(0.2 * h)
        self.patch_region = (start_x, start_y, 
                           start_x + self.m * self.block_size,
                           start_y + self.m * self.block_size)
        
        # Initial position of the agent in the grid
        self.current_pos = (0, 0)
        
        # Get original confidence
        self.original_confidence = self.run_detector(self.current_image)
        self.current_confidence = self.original_confidence
        self.step_count = 0
        
        # Return initial state
        return self._get_state()
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: Integer representing movement direction and grayscale choice
            
        Returns:
            next_state, reward, done, info
        """
        self.step_count += 1
        done = False
        
        # Decode action: 0-3 for directions with grayscale[0], 4-7 for directions with grayscale[1], etc.
        num_directions = 4  # up, down, left, right
        direction = action % num_directions
        gray_idx = action // num_directions
        
        # Check if gray_idx is valid
        if gray_idx >= len(self.grayscale_values):
            gray_idx = 0  # Default to first grayscale value
        
        gray_value = self.grayscale_values[gray_idx]
        
        # Calculate new position based on direction
        old_pos = self.current_pos
        if direction == 0:  # up
            new_pos = (old_pos[0], max(0, old_pos[1] - 1))
        elif direction == 1:  # down
            new_pos = (old_pos[0], min(self.m - 1, old_pos[1] + 1))
        elif direction == 2:  # left
            new_pos = (max(0, old_pos[0] - 1), old_pos[1])
        elif direction == 3:  # right
            new_pos = (min(self.m - 1, old_pos[0] + 1), old_pos[1])
        
        # Update current position
        self.current_pos = new_pos
        
        # Add new block to patch_blocks if not already in list
        block_info = (new_pos[0], new_pos[1], gray_value)
        if block_info not in self.patch_blocks:
            self.patch_blocks.append(block_info)
            self.grid[new_pos[1], new_pos[0]] = gray_value
        
        # Update the image with the new patch
        image_with_patch = self._apply_patch_to_image(self.current_image.copy())
        
        # Get new confidence
        prev_confidence = self.current_confidence
        self.current_confidence = self.run_detector(image_with_patch)
        
        # Calculate reward (positive if confidence decreases)
        reward = np.sign(prev_confidence - self.current_confidence)
        
        # Check if done (confidence below threshold or max steps reached)
        if self.current_confidence < 0.5 or self.step_count >= self.m * self.m:
            done = True
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'current_confidence': self.current_confidence,
            'original_confidence': self.original_confidence,
            'num_blocks': len(self.patch_blocks)
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """
        Get the current state representation.
        
        Returns:
            State vector
        """
        # State includes current position and grid state
        pos_x_norm = self.current_pos[0] / (self.m - 1)
        pos_y_norm = self.current_pos[1] / (self.m - 1)
        grid_flat = self.grid.flatten()
        
        # Concatenate position and grid state
        state = np.concatenate([[pos_x_norm, pos_y_norm], grid_flat])
        return state
    
    def _apply_patch_to_image(self, image):
        """
        Apply the current patch blocks to the image.
        
        Args:
            image: Image to apply patch to
            
        Returns:
            Image with patch applied
        """
        for block in self.patch_blocks:
            x, y, gray_value = block
            
            # Convert normalized gray_value [0,1] to absolute [0,255]
            gray_abs = int(gray_value * 255)
            
            # Calculate pixel coordinates
            px = self.patch_region[0] + x * self.block_size
            py = self.patch_region[1] + y * self.block_size
            
            # Draw block on image
            image[py:py+self.block_size, px:px+self.block_size] = gray_abs
            
        return image
    
    def run_detector(self, image):
        """
        Run the object detector on the image and return confidence.
        
        Args:
            image: Image to run detection on
            
        Returns:
            Confidence score
        """
        # This is a placeholder - in a real implementation, you would call your detector
        # Return a simulated confidence score based on the detector's output
        results = self.detector(image)
        
        # Extract confidence for the object at the bbox location
        # This is a simplified version - actual implementation would depend on detector's output format
        confidence = self.detector.get_confidence_at_bbox(image, self.bbox)
        
        return confidence
    
    def render(self):
        """
        Render the current state of the environment.
        
        Returns:
            Image with patch visualization
        """
        image = self.current_image.copy()
        image_with_patch = self._apply_patch_to_image(image)
        
        # Draw bounding box
        cv2.rectangle(image_with_patch, 
                     (self.bbox[0], self.bbox[1]), 
                     (self.bbox[2], self.bbox[3]), 
                     (0, 255, 0), 2)
        
        # Draw patch region
        cv2.rectangle(image_with_patch, 
                     (self.patch_region[0], self.patch_region[1]), 
                     (self.patch_region[2], self.patch_region[3]), 
                     (255, 0, 0), 1)
        
        return image_with_patch

# Mock detector class for testing
class YOLOv5Detector:
    def __init__(self, model_path='yolov5s.pt'):
        self.model = YOLO(model_path)  # Load the YOLOv5 model
    
    def __call__(self, image):
        results = self.model(image)  # Run inference
        return results
    
    def get_confidence_at_bbox(self, image, bbox):
        results = self.model(image)[0]
        max_conf = 0
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if self._iou([x1, y1, x2, y2], bbox) > 0.5:
                max_conf = max(max_conf, conf)
        return max_conf
    
    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area else 0
def train_rlp(detector, train_data, num_episodes=600, max_steps=25):
    """
    Train the RLP agent to generate adversarial patches.
    
    Args:
        detector: Object detector model
        train_data: List of (image, bbox) tuples
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        
    Returns:
        Trained agent
    """
    # Set parameters
    grid_size = (5, 5)  # m=5 from the paper
    grayscale_values = [0.1, 0.3]  # g1=0.1, g2=0.3 from the paper
    
    # Calculate state and action sizes
    state_size = 2 + grid_size[0] * grid_size[1]  # Position (2) + grid state
    action_size = 4 * len(grayscale_values)  # 4 directions * number of grayscale values
    
    # Initialize environment and agent
    env = RLPEnvironment(grid_size, grayscale_values, detector)
    agent = RLPAgent(state_size, action_size, grid_size, grayscale_values, max_steps, detector)
    
    # Training loop
    for episode in range(num_episodes):
        # Sample a random training example
        image, bbox = random.choice(train_data)
        
        # Reset environment
        state = env.reset(image, bbox)
        total_reward = 0
        
        for t in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition in memory
            agent.memory.push(state, action, next_state, reward, done)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            
            # Optimize model
            loss = agent.optimize_model()
            
            if done:
                break
        
        # Update target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Update epsilon
        agent.update_epsilon(episode)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, "
                  f"Epsilon: {agent.epsilon:.2f}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Final Confidence: {info['current_confidence']:.4f}")
    
    return agent

def test_rlp(agent, detector, test_data):
    """
    Test the trained RLP agent on test data.
    
    Args:
        agent: Trained RLP agent
        detector: Object detector model
        test_data: List of (image, bbox) tuples
        
    Returns:
        Dictionary with results
    """
    # Initialize environment
    grid_size = (5, 5)
    grayscale_values = [0.1, 0.3]
    env = RLPEnvironment(grid_size, grayscale_values, detector)
    
    results = {
        'original_confidences': [],
        'attack_confidences': [],
        'success_rate': 0,
    }
    
    total_samples = len(test_data)
    success_count = 0
    
    for i, (image, bbox) in enumerate(test_data):
        # Reset environment
        state = env.reset(image, bbox)
        original_confidence = env.original_confidence
        
        # Apply the learned policy without exploration
        for t in range(agent.max_steps):
            action = agent.select_action(state, epsilon=0)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break
        
        attack_confidence = env.current_confidence
        success = attack_confidence < 0.5
        
        if success:
            success_count += 1
        
        results['original_confidences'].append(original_confidence)
        results['attack_confidences'].append(attack_confidence)
        
        if i % 10 == 0:
            print(f"Testing sample {i+1}/{total_samples}, "
                  f"Original conf: {original_confidence:.4f}, "
                  f"Attack conf: {attack_confidence:.4f}, "
                  f"Success: {success}")
    
    results['success_rate'] = success_count / total_samples
    print(f"Overall attack success rate: {results['success_rate']:.4f}")
    
    return results

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def generate_bboxes(images, detector):
    bboxes = []
    for img in images:
        results = detector(img)
        if results and results[0].boxes:
            bbox = results[0].boxes.xyxy[0].tolist()
            bboxes.append(bbox)
        else:
            bboxes.append([50, 50, 200, 200])  # Default bbox if no detection
    return bboxes

def main():
    # This is a mock implementation - in a real scenario, you would:
    # 1. Load your actual object detector model
    # 2. Load your actual dataset
    
    detector = YOLOv5Detector()
    train_folder = 'train_RL_FLIR_Data'
    test_folder = 'test_RL_FLIR_Data'
    
    train_images = load_images_from_folder(train_folder)
    test_images = load_images_from_folder(test_folder)
    
    train_bboxes = generate_bboxes(train_images, detector)
    test_bboxes = generate_bboxes(test_images, detector)
    
    train_data = list(zip(train_images, train_bboxes))
    test_data = list(zip(test_images, test_bboxes))
    
    print("Training RLP agent...")
    agent = train_rlp(detector, train_data, num_episodes=600, max_steps=25)
    
    print("Testing RLP agent...")
    results = test_rlp(agent, detector, test_data)
    
    print(f"Attack success rate: {results['success_rate']:.4f}")
    print("Training and testing complete!")
if __name__ == "__main__":
    main()