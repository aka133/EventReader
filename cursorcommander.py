import requests
import json
import time
from typing import Dict, List, Tuple, Optional
import pyautogui
import cv2
import base64
import io
from PIL import Image
import numpy as np
import traceback
import re

class CursorCommander:
    def __init__(self, 
                 llm_model: str,
                 screen_width: int,
                 screen_height: int,
                 debug_mode: bool = False):
        """
        Handles cursor command generation and execution.
        
        Args:
            llm_model: Model to use for command generation
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            debug_mode: Enable detailed logging
        """
        self.llm_model = llm_model
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.debug_mode = debug_mode
        self.last_cursor_pos: Tuple[int, int] = (0, 0)
        self.command_history: List[Dict] = []

    def _encode_frame(self, frame):
        """Convert frame to base64 for LLM"""
        try:
            if not isinstance(frame, np.ndarray):
                print(f"[DEBUG] Invalid frame type: {type(frame)}")
                return None
            
            # Add shape validation
            if len(frame.shape) != 3 or frame.shape[2] not in [3, 4]:
                print(f"[DEBUG] Invalid frame shape: {frame.shape}")
                return None
                
            # Convert BGRA to RGB if needed
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            
            # Keep existing conversion logic
            img = Image.fromarray(frame)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG", optimize=True, quality=85)
            encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Validate encoded data
            if not encoded:
                print("[ERROR] Failed to encode image")
                return None
                
            return encoded
            
        except Exception as e:
            print(f"[ERROR] Frame encoding failed: {e}")
            traceback.print_exc()
            return None

    def generate_commands(self, context, goal):
        try:
            # Keep existing frame handling
            frame = context.get('current_frame')
            if frame is None:
                print("[ERROR] No frame available for visual analysis")
                return []

            # Add debug logging without changing logic
            print(f"[DEBUG] Frame type: {type(frame)}, Shape: {frame.shape if hasattr(frame, 'shape') else 'No shape'}")
            
            # Keep existing active_regions_info logic
            active_regions_info = []
            for region_key, data in context.get('active_regions', {}).items():
                if data.get('events'):
                    recent_intensity = sum(e['intensity'] for e in data['events']) / len(data['events'])
                    active_regions_info.append(
                        f"Region {region_key}: {len(data['events'])} recent events, "
                        f"average intensity {recent_intensity:.2f}"
                    )

            # Ensure frame is encoded properly
            encoded_frame = self._encode_frame(frame)
            if not encoded_frame:
                print("[ERROR] Frame encoding failed")
                return []

            # Keep existing prompt structure but ensure images are properly formatted
            prompt = {
                "model": self.llm_model,
                "prompt": f"""Task: Generate cursor commands to {goal}

Screen Activity:
{chr(10).join(active_regions_info)}

Rules:
1. Output ONLY a JSON array
2. Use ONLY these commands:
   - MOVE [x, y]
   - CLICK []
   - DOUBLECLICK []
   - RIGHTCLICK []

Screen: {self.screen_width}x{self.screen_height}
Cursor: {self.last_cursor_pos}""",
                "images": [encoded_frame]  # Ensure it's a list of base64 strings
            }

            # Add request debugging
            print(f"[DEBUG] Request contains images: {len(prompt['images'])}")
            print(f"[DEBUG] Image data starts with: {prompt['images'][0][:50]}...")

            # Keep existing request logic
            response = requests.post(
                'https://z8ezes3rip4j66-8000.proxy.runpod.net/analyze',
                json=prompt,
                verify=False,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[ERROR] Server response: {response.status_code} - {response.text}")
                return []
                
            return self._parse_commands(response.json()['response'])
            
        except Exception as e:
            print(f"[ERROR] Command generation error: {str(e)}")
            traceback.print_exc()
            return []

    def _parse_commands(self, response: str) -> List[Dict]:
        """Parse LLM response with fallbacks"""
        try:
            # Try to find JSON array in response
            match = re.search(r'\[(.*?)\]', response.replace('\n', ''), re.DOTALL)
            if match:
                commands = json.loads(f"[{match.group(1)}]")
                if isinstance(commands, list):
                    return [
                        {
                            'type': cmd['type'].upper(),
                            'parameters': cmd.get('parameters', []),
                            'purpose': cmd.get('purpose', ''),
                            'timestamp': time.time()
                        }
                        for cmd in commands
                    ]
        except json.JSONDecodeError:
            print("[ERROR] Failed to parse JSON response")
            if self.debug_mode:
                print("[DEBUG] JSON parse failed, trying legacy parsing")
            return self._parse_commands_legacy(response)
                
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Command parsing error: {e}")
                traceback.print_exc()
            return []

    def _parse_commands_legacy(self, response: str) -> List[Dict]:
        """Legacy line-by-line command parsing"""
        commands = []
        for line in response.split('\n'):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            try:
                # Split command and comment
                parts = line.split('//')
                command_part = parts[0].strip()
                comment = parts[1].strip() if len(parts) > 1 else ""
                
                # Parse command and parameters
                tokens = command_part.split()
                command_type = tokens[0].upper()
                parameters = tokens[1:] if len(tokens) > 1 else []
                
                command = {
                    'type': command_type,
                    'parameters': parameters,
                    'purpose': comment,
                    'timestamp': time.time()
                }
                
                # Update cursor position for MOVE commands
                if command_type == 'MOVE' and len(parameters) >= 2:
                    self.last_cursor_pos = (int(parameters[0]), int(parameters[1]))
                
                commands.append(command)
                
            except Exception as e:
                if self.debug_mode:
                    print(f"[DEBUG] Failed to parse command '{line}': {e}")
                continue
        
        return commands

    def validate_commands(self, commands: List[Dict]) -> bool:
        """Validate generated commands for safety and feasibility"""
        try:
            for cmd in commands:
                # Validate command structure
                if not all(key in cmd for key in ['type', 'parameters', 'purpose']):
                    if self.debug_mode:
                        print(f"[DEBUG] Invalid command structure: {cmd}")
                    return False
                
                # Validate coordinates are within screen bounds
                if cmd['type'] in ['MOVE', 'DRAG']:
                    coords = cmd['parameters']
                    for i in range(0, len(coords), 2):
                        x, y = int(coords[i]), int(coords[i+1])
                        if not (0 <= x <= self.screen_width and 0 <= y <= self.screen_height):
                            if self.debug_mode:
                                print(f"[DEBUG] Coordinates ({x}, {y}) out of bounds")
                            return False
                
            return True
            
        except Exception as e:
            print(f"[ERROR] Command validation error: {e}")
            return False
        
    def execute_command(self, command: Dict) -> bool:
        """Execute a single cursor command"""
        try:
            cmd_type = command['type'].upper()
            params = command['parameters']
            
            if self.debug_mode:
                print(f"[DEBUG] Executing: {cmd_type} {params}")
            
            if not self.validate_commands([command]):
                return False
            
            if cmd_type == 'MOVE':
                x, y = int(params[0]), int(params[1])
                pyautogui.moveTo(x, y, duration=0.2)
                self.last_cursor_pos = (x, y)
                
            elif cmd_type == 'CLICK':
                pyautogui.click()
                
            elif cmd_type == 'DOUBLECLICK':
                pyautogui.doubleClick()
                
            elif cmd_type == 'RIGHTCLICK':
                pyautogui.rightClick()
                
            elif cmd_type == 'DRAG':
                start_x, start_y = int(params[0]), int(params[1])
                end_x, end_y = int(params[2]), int(params[3])
                pyautogui.dragTo(end_x, end_y, duration=0.5)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Command execution failed: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return False

def main():
    """Test the CursorCommander"""
    commander = CursorCommander(
        llm_model="llama2",
        screen_width=1440,
        screen_height=900,
        debug_mode=True
    )
    
    # Test context
    test_context = {
        'screen_resolution': '1440x900',
        'active_regions': {
            '2,3': {'count': 15, 'screen_pos': (450, 280)},
            '4,5': {'count': 8, 'screen_pos': (800, 400)}
        },
        'recent_activities': [
            {'type': 'click', 'region': '2,3', 'intensity': 0.8},
            {'type': 'hover', 'region': '4,5', 'intensity': 0.6}
        ],
        'current_frame': None  # Would contain base64 encoded image
    }
    
    # Test goal
    test_goal = "Click the search box in the top-right corner"
    
    # Generate commands
    commands = commander.generate_commands(test_context, test_goal)
    
    # Print results
    print("\nGenerated Commands:")
    for cmd in commands:
        print(f"{cmd['type']} {' '.join(map(str, cmd['parameters']))} // {cmd['purpose']}")

if __name__ == "__main__":
    main()