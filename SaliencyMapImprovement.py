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
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image

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

class CAMRLPAgent:
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
        
        # Setup CAM
        if hasattr(detector_model, 'model'):
            self.cam_model = detector_model.model
            # Usually the last convolutional layer is most informative for CAM
            if hasattr(self.cam_model, 'model') and hasattr(self.cam_model.model, 'model'):
                self.target_layers = [self.cam_model.model.model[-2]]
            else:
                # Fallback target layer
                self.target_layers = [list(self.cam_model.modules())[-2]]
            self.cam = EigenCAM(self.cam_model, self.target_layers)
        else:
            self.cam_model = detector_model
            # Fallback target layer
            self.target_layers = [list(self.cam_model.modules())[-2]]
            self.cam = EigenCAM(self.cam_model, self.target_layers)
        
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
    
    def compute_cam(self, image):
        """Compute the Class Activation Map for the image"""
        
        # Convert image to tensor
        img = np.array(image)
        if len(img.shape) == 2:  # If grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize image for model input
        img_float = np.float32(img) / 255
        transform = transforms.ToTensor()
        tensor = transform(img_float).unsqueeze(0)
        
        # Move tensor to same device as model
        device = next(self.cam_model.parameters()).device
        tensor = tensor.to(device)
        
        # Generate CAM
        # grayscale_cam = self.cam(tensor)[0, :]
        cam_output = self.cam(tensor)
        if isinstance(cam_output, tuple):
            cam_output = cam_output[0]
        grayscale_cam = cam_output[0, :]
        return grayscale_cam

class CAMRLPEnvironment:
    def __init__(self, grid_size, grayscale_values, obj_detector):
        """
        Initialize the RLP environment with CAM support.
        
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
        self.cam_map = None  # Store the CAM for the current image
        
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
        
        # Compute CAM for the current image
        self.cam_map = self.compute_cam(image, bbox)
        
        # Use CAM to find the most important region to start (highest activation)
        cam_grid = self.cam_to_grid(self.cam_map)
        start_pos = np.unravel_index(np.argmax(cam_grid), cam_grid.shape)
        self.current_pos = (start_pos[1], start_pos[0])  # (x, y) format
        
        # Get original confidence
        self.original_confidence = self.run_detector(self.current_image)
        self.current_confidence = self.original_confidence
        self.step_count = 0
        
        # Return initial state
        return self._get_state()
    
    def compute_cam(self, image, bbox):
        """
        Compute Class Activation Map for the image.
        
        Args:
            image: Input image
            bbox: Bounding box of interest
        
        Returns:
            CAM map normalized to [0,1]
        """
        # This is a placeholder - integrate with your CAM implementation
        if hasattr(self.detector, 'compute_cam'):
            cam_map = self.detector.compute_cam(image)
        else:
            # Use a pre-computed CAM or a dummy one
            cam_map = np.ones((image.shape[0], image.shape[1]))
            
            # Focus only on the bbox region
            cam_map[:, :] = 0.1
            x1, y1, x2, y2 = map(int, bbox)
            cam_map[y1:y2, x1:x2] = 0.8
        
        return cam_map
    
    def cam_to_grid(self, cam_map):
        """
        Map the CAM to the grid space.
        
        Args:
            cam_map: Class activation map
                
        Returns:
            Grid representation of CAM
        """
        # Ensure all indices are integers
        x1, y1, x2, y2 = map(int, self.patch_region)
        
        # Make sure indices are within bounds
        height, width = cam_map.shape
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(x1+1, min(x2, width))
        y2 = max(y1+1, min(y2, height))
        
        region_cam = cam_map[y1:y2, x1:x2]
        
        # Resize to grid dimensions
        grid_cam = cv2.resize(region_cam, (self.m, self.m))
        return grid_cam
    
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
        
        # Calculate reward (considering both confidence decrease and CAM value)
        # Higher reward if patch is placed on important regions according to CAM
        conf_reward = prev_confidence - self.current_confidence
        cam_grid = self.cam_to_grid(self.cam_map)
        cam_reward = cam_grid[new_pos[1], new_pos[0]] / 5.0  # Normalize CAM reward
        
        # Combined reward: confidence decrease + CAM importance
        reward = conf_reward + cam_reward
        
        # Check if done (confidence below threshold or max steps reached)
        if self.current_confidence < 0.5 or self.step_count >= self.m * self.m:
            done = True
            
            # Bonus reward for successfully fooling the detector
            if self.current_confidence < 0.5:
                reward += 5.0
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'current_confidence': self.current_confidence,
            'original_confidence': self.original_confidence,
            'num_blocks': len(self.patch_blocks),
            'cam_value': cam_grid[new_pos[1], new_pos[0]]
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """
        Get the current state representation, including CAM information.
        
        Returns:
            State vector
        """
        # Convert current position to normalized coordinates
        pos_x_norm = self.current_pos[0] / (self.m - 1)
        pos_y_norm = self.current_pos[1] / (self.m - 1)
        
        # Include grid state
        grid_flat = self.grid.flatten()
        
        # Include CAM information for the grid
        cam_grid = self.cam_to_grid(self.cam_map)
        cam_flat = cam_grid.flatten()
        
        # Concatenate position, grid state, and CAM information
        state = np.concatenate([[pos_x_norm, pos_y_norm], grid_flat, cam_flat])
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
            
            # Calculate pixel coordinates with explicit integer casting
            px_start = int(self.patch_region[0] + x * self.block_size)
            py_start = int(self.patch_region[1] + y * self.block_size)
            px_end = int(px_start + self.block_size)
            py_end = int(py_start + self.block_size)
            
            # Draw block on image using integer indices
            image[py_start:py_end, px_start:px_end] = gray_abs
            
        return image
    
    def run_detector(self, image):
        """
        Run the object detector on the image and return confidence.
        
        Args:
            image: Image to run detection on
            
        Returns:
            Confidence score
        """
        results = self.detector(image)
        
        # Extract confidence for the object at the bbox location
        confidence = self.detector.get_confidence_at_bbox(image, self.bbox)
        
        return confidence
    
    def render(self, show_cam=True):
        """
        Render the current state of the environment.
        
        Args:
            show_cam: Whether to overlay CAM visualization
            
        Returns:
            Image with patch and CAM visualization
        """
        image = self.current_image.copy()
        image_with_patch = self._apply_patch_to_image(image)
        
        # Draw bounding box - ensure proper integer coordinates
        x1, y1, x2, y2 = map(int, self.bbox)
        cv2.rectangle(image_with_patch, 
                     (x1, y1), 
                     (x2, y2), 
                     (0, 255, 0), 2)
        
        # Draw patch region - ensure proper integer coordinates
        px1, py1, px2, py2 = map(int, self.patch_region)
        cv2.rectangle(image_with_patch, 
                     (px1, py1), 
                     (px2, py2), 
                     (255, 0, 0), 1)
        
        # If requested, overlay CAM visualization
        if show_cam:
            # Convert grayscale CAM to heatmap
            img_float = np.float32(image) / 255
            cam_image = show_cam_on_image(img_float, self.cam_map, use_rgb=True)
            
            # Blend with the original image
            alpha = 0.5
            cam_overlay = cv2.addWeighted(image_with_patch, 1-alpha, 
                                         np.uint8(cam_image*255), alpha, 0)
            return cam_overlay
        
        return image_with_patch

class YOLOv5Detector:
    def __init__(self, model_path='yolov5s.pt'):
        # self.model = YOLO(model_path)  # Load the YOLOv5 model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)  # Explicitly move model to device
        self.target_layers = [self.model.model.model[-2]]  # Usually last conv layer
        self.cam = EigenCAM(self.model.model, self.target_layers)
    
        
        # Setup CAM
        self.target_layers = [self.model.model.model[-2]]  # Usually last conv layer
        self.cam = EigenCAM(self.model.model, self.target_layers)
    
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
    
    def compute_cam(self, image):
        """Compute Class Activation Map for the image compatible with YOLOv8"""
        # Convert image to numpy array
        img = np.array(image)
        if len(img.shape) == 2:  # If grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize image for model input
        img_float = np.float32(img) / 255
        
        # Create a custom hook to get activations
        activations = []
        def hook_fn(module, input, output):
            activations.clear()  # Clear previous activations
            activations.append(output.detach())
            
        # Get the target layer (usually the last convolutional layer)
        target_layer = self.model.model.model[-2]
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
        # Run inference
        with torch.no_grad():
            # Process the image using YOLO's preprocessing
            results = self.model(img)
            
        # Remove the hook
        hook_handle.remove()
        
        # Generate a simple CAM using the activations
        # For simplicity, we'll use the mean of activations across channels
        if activations:
            # Get the activation map
            activation = activations[0]
            cam = activation.mean(dim=1).squeeze().cpu().numpy()
            
            # Resize to match input image dimensions
            cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
            
            # Normalize to [0, 1]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
            activations.clear()
            return cam
        else:
            # Fallback to a uniform CAM
            activations.clear()
            return np.ones((img.shape[0], img.shape[1]))
            
def train_cam_rlp(detector, train_data, num_episodes=600, max_steps=25):
    """
    Train the RLP agent with CAM guidance to generate adversarial patches.
    
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
    # State size now includes CAM information for each grid cell
    state_size = 2 + 2 * grid_size[0] * grid_size[1]  # Position (2) + grid state + CAM grid
    action_size = 4 * len(grayscale_values)  # 4 directions * number of grayscale values
    
    # Initialize environment and agent
    env = CAMRLPEnvironment(grid_size, grayscale_values, detector)
    agent = CAMRLPAgent(state_size, action_size, grid_size, grayscale_values, max_steps, detector)
    
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
                  f"Final Confidence: {info['current_confidence']:.4f}, "
                  f"CAM Value: {info['cam_value']:.4f}")
            
            # Save visualization
            # if episode % 50 == 0:
            #     vis_img = env.render(show_cam=True)
                # cv2.imwrite(f"cam_rlp_episode_{episode}.png", vis_img)
    
    return agent

def test_cam_rlp(agent, detector, test_data):
    """
    Test the trained CAM-guided RLP agent on test data.
    
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
    env = CAMRLPEnvironment(grid_size, grayscale_values, detector)
    
    results = {
        'original_confidences': [],
        'attack_confidences': [],
        'success_rate': 0,
        'cam_values': [],
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
        results['cam_values'].append(info['cam_value'])
        
        if i % 10 == 0:
            print(f"Testing sample {i+1}/{total_samples}, "
                  f"Original conf: {original_confidence:.4f}, "
                  f"Attack conf: {attack_confidence:.4f}, "
                  f"Success: {success}, "
                  f"CAM Value: {info['cam_value']:.4f}")
            
            # Save visualization
            vis_img = env.render(show_cam=True)
            cv2.imwrite(f"cam_rlp_test_{i}.png", vis_img)
    
    results['success_rate'] = success_count / total_samples
    print(f"Overall attack success rate: {results['success_rate']:.4f}")
    
    return results

def main():
    detector = YOLOv5Detector()
    train_folder = 'train_RL_FLIR_Data'
    test_folder = 'test_RL_FLIR_Data'
    
    # Load images and generate bboxes
    train_images = load_images_from_folder(train_folder)
    test_images = load_images_from_folder(test_folder)
    
    train_bboxes = generate_bboxes(train_images, detector)
    test_bboxes = generate_bboxes(test_images, detector)
    
    train_data = list(zip(train_images, train_bboxes))
    test_data = list(zip(test_images, test_bboxes))
    
    print("Training CAM-guided RLP agent...")
    agent = train_cam_rlp(detector, train_data, num_episodes=600, max_steps=25)
    
    print("Testing CAM-guided RLP agent...")
    results = test_cam_rlp(agent, detector, test_data)
    
    print(f"Attack success rate: {results['success_rate']:.4f}")
    print("Training and testing complete!")

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

if __name__ == "__main__":
    main()
