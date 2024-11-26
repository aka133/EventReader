import torch
import numpy as np
from collections import deque
import time
from eventreader_leg4 import EventDrivenScreenReader
from cursorcommander import CursorCommander
import requests
import json
from PIL import Image
import io
import base64
import cv2
import os
import pyautogui
import threading
import traceback
import psutil

from queue import Queue

class VisualActivityAnalyzer:
    def __init__(self, 
                 llm_model: str = "bakllava",
                 debug_mode: bool = True,
                 context_update_interval: float = 5.0,
                 activity_threshold: int = 2):
        """
        Main analyzer that orchestrates screen reading, context generation, and cursor control.
        
        Args:
            llm_model: Model to use for visual analysis
            debug_mode: Enable detailed logging
            context_update_interval: Seconds between context updates
            activity_threshold: Minimum activity count to consider region active
        """
        print("[INFO] Initializing Visual Activity Analyzer...")
        
        # Core settings
        self.llm_model = llm_model
        self.debug_mode = debug_mode
        self.context_update_interval = context_update_interval
        self.activity_threshold = activity_threshold
        self.intensity_threshold = 0.1
        self.grid_size = 8
        
        # Initialize event queue first
        self.event_queue = Queue()
        
        # Add these new attributes for request management
        self.pending_llm_request = False
        self.min_context_interval = 25.0  # Minimum seconds between requests
        self.last_context_update = time.time()
        
        # Activity tracking
        self.active_regions = {}
        self.activity_buffer = deque(maxlen=100)
        self.dirty_regions = set()
        self.last_update = time.time()
        self.last_cleanup = time.time()  # Missing attribute causing error
        self.cleanup_interval = 5
        
        # Context management
        self.event_buffer = deque(maxlen=1000)
        self.context_buffer = deque(maxlen=10)
        self.activity_summary = deque(maxlen=50)
        self.current_context = ""
        self.last_context_update = time.time()
        self.running = True
        
        # Add rate limiting for context updates
        self.min_events_for_update = 20  # Minimum events before updating context
        self.event_count_since_update = 0
        
        # Add bounds to prevent memory leaks
        self.activity_buffer = deque(maxlen=100)  # Limit activity buffer size
        self.event_buffer = deque(maxlen=1000)    # Limit event buffer size
        self.context_buffer = deque(maxlen=10)    # Limit context history
        self.activity_summary = deque(maxlen=50)  # Limit activity summary
        
        # Add rate limiting
        self.min_event_interval = 0.05  # 20 events per second max
        self.last_event_time = 0
        self.last_visualization_time = 0
        self.visualization_interval = 0.5  # 2 FPS max for visualization
        
        # Initialize components
        print("[INFO] Setting up screen reader...")
        try:
            self.reader = EventDrivenScreenReader(
                event_callback=self._handle_screen_event,
            )
            print("[INFO] Screen reader initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize screen reader: {e}")
            raise
        
        print("[INFO] Setting up cursor commander...")
        self.cursor_commander = CursorCommander(
            llm_model=llm_model,
            screen_width=self.reader.screen_width,
            screen_height=self.reader.screen_height,
            debug_mode=debug_mode
        )
        # Check inference server connection
        print("[INFO] Checking inference server connection...")
        try:
            test_data = {
                "images": ["test"],
                "prompt": "test"
            }
            response = requests.post(
                'https://z8ezes3rip4j66-8000.proxy.runpod.net/analyze',
                json=test_data,
                verify=False,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            print(f"[DEBUG] Response status: {response.status_code}")
            print(f"[DEBUG] Response text: {response.text[:200]}")
            
            if response.status_code == 200:
                print("[INFO] Successfully connected to inference server")
            else:
                print("[WARN] Inference server responded with status:", response.status_code)
        except requests.exceptions.ConnectionError as e:
            print("[WARN] Could not connect to inference server:", str(e))
            print("Continuing without inference server - some features may be limited")
        except Exception as e:
            print(f"[WARN] Unexpected error checking inference server: {e}")
        
        print("[INFO] Initialization complete!")

        # Add debug directory for visualizations
        self.debug_dir = "debug_frames"
        if debug_mode and not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        # Replace polling-based processing thread with event-driven approach
        self.context_update_event = threading.Event()
        self.cleanup_event = threading.Event()
        self.processing_thread = threading.Thread(target=self._event_loop, daemon=True)
        self.processing_thread.start()

    def _event_loop(self):
        """Event-driven processing loop"""
        while self.running:
            # Wait for events with timeout for periodic tasks
            timeout = min(
                self.context_update_interval - (time.time() - self.last_context_update),
                self.cleanup_interval - (time.time() - self.last_cleanup)
            )
            if timeout > 0:
                self.context_update_event.wait(timeout)
                self.context_update_event.clear()
            
            current_time = time.time()
            
            # Handle periodic cleanup
            if current_time - self.last_cleanup >= self.cleanup_interval:
                self._cleanup_resources()
                self.last_cleanup = current_time
            
            # Handle context updates
            if (current_time - self.last_context_update >= self.context_update_interval and 
                len(self.active_regions) > 0):
                self._update_context()

    def _handle_screen_event(self, x: int, y: int, intensity: float):
        """Process incoming screen events"""
        try:
            current_time = time.time()
            
            # Convert to grid coordinates
            region_x = min(int((x / self.reader.screen_width) * self.grid_size), 
                          self.grid_size - 1)
            region_y = min(int((y / self.reader.screen_height) * self.grid_size), 
                          self.grid_size - 1)
            region_key = f"{region_x},{region_y}"
            
            # Update region data
            if region_key not in self.active_regions:
                self.active_regions[region_key] = {
                    'count': 0,
                    'last_intensity': 0,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'screen_pos': (x, y)
                }
            
            region = self.active_regions[region_key]
            region['count'] += 1
            region['last_intensity'] = max(intensity, region['last_intensity'])
            region['last_seen'] = current_time
            region['screen_pos'] = (x, y)
            
            # Create activity record
            activity = {
                'type': 'motion',
                'region': region_key,
                'screen_pos': (x, y),
                'intensity': intensity,
                'time': current_time
            }
            
            # Update activity summary
            self.activity_summary.append(activity)
            if len(self.activity_summary) > 50:  # Keep last 50 activities
                self.activity_summary = self.activity_summary[-50:]
            
            self.event_count_since_update += 1
            
            # Signal context update if needed
            if (self.event_count_since_update >= self.min_events_for_update and 
                current_time - self.last_context_update >= self.context_update_interval):
                if self.debug_mode:
                    print("[DEBUG] Triggering context update")
                self._update_context()
                self.event_count_since_update = 0
                
        except Exception as e:
            print(f"[ERROR] Event handling error: {e}")
            if self.debug_mode:
                traceback.print_exc()

    def _classify_activity(self, intensity: float, count: int) -> str:
        """Classify activity based on intensity and frequency"""
        if count > 20 and intensity > 0.55:
            return "Sustained Rapid Activity"
        elif intensity > 0.8:
            return "Rapid Change"
        elif count > 10:
            return "Repeated Activity"
        elif intensity > 0.2:
            return "Moderate Activity"
        return "Subtle Movement"

    def _cleanup_old_regions(self, current_time):
        """Clean up old activity regions"""
        regions_to_remove = []
        for region, data in self.active_regions.items():
            if current_time - data['last_seen'] > self.cleanup_interval:
                regions_to_remove.append(region)
        
        for region in regions_to_remove:
            del self.active_regions[region]

    def _generate_activity_visualization(self):
        """Generate visualization of current activity"""
        try:
            # Get raw frame data and ensure it's valid
            frame_data = self.reader.get_current_frame()
            if frame_data is None:
                print("[DEBUG] No frame data received")
                return None
            
            frame, frame_time = frame_data
            
            # Debug the incoming frame
            if self.debug_mode:
                print(f"[DEBUG] Raw frame before processing:")
                print(f"  Shape: {frame.shape}")
                print(f"  Dtype: {frame.dtype}")
                print(f"  Range: {frame.min()}-{frame.max()}")
                print(f"  Mean: {frame.mean()}")
            
            # Create visualization base (ensure proper copy)
            vis = np.ascontiguousarray(frame, dtype=np.uint8)
            
            # Ensure proper color conversion
            if len(vis.shape) == 3 and vis.shape[2] == 3:
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            
            # Verify the output
            if vis.min() == 0 and vis.max() == 0:
                print("[ERROR] Visualization frame is black")
                return None
            
            return vis, frame_time
            
        except Exception as e:
            print(f"[ERROR] Visualization failed: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return None

    def _process_frame(self, frame):
        """Process a single frame without token conversion"""
        try:
            if frame is None:
                return
            
            # Process the frame directly as numpy array
            #if self.debug_mode:
            #    timestamp = time.strftime("%Y%m%d_%H%M%S")
            #    debug_path = f"debug_frame_{timestamp}.png"
            #    cv2.imwrite(debug_path, frame)
            #    print(f"[DEBUG] Saved debug frame to {debug_path}")
            
        except Exception as e:
            print(f"[ERROR] Frame processing error: {e}")
            if self.debug_mode:
                traceback.print_exc()

    def _update_context(self):
        try:
            current_time = time.time()
            
            # More aggressive rate limiting
            if self.pending_llm_request:
                return
            if current_time - self.last_context_update < self.min_context_interval:
                return
            
            if not self.active_regions:
                return

            understanding = self.get_current_understanding()
            
            # Skip if no significant changes
            if len(understanding.get('recent_activities', [])) < 5:
                print("[DEBUG] Skipping update - not enough activity")
                return
            
            prompt = self._build_prompt(understanding)
            
            self.pending_llm_request = True
            print(f"[DEBUG] Starting LLM request at {time.strftime('%H:%M:%S')}")
            
            threading.Thread(
                target=self._send_llm_request,
                args=(prompt, current_time),
                daemon=True
            ).start()
            
        except Exception as e:
            print(f"[ERROR] Context update error: {e}")
            self.pending_llm_request = False

    def _build_prompt(self, understanding):
        """Build a more structured and clear prompt for the LLM"""
        try:
            screen_width = self.reader.screen_width
            screen_height = self.reader.screen_height
            
            # Get image data safely
            image_data = understanding.get('current_frame')
            if isinstance(image_data, (str, bytes)) and image_data:
                prompt = {
                    "model": self.llm_model,
                    "prompt": f"""<image>
                    Analyze this screen activity data:
                    Screen Size: {screen_width}x{screen_height}

                    1. Visual Layout:
                        - Describe the exact position and content of each window/application
                        - Note any UI elements like buttons, menus, or interactive areas
                        - Identify text content and its location
                        - Describe any media content (images, videos) and their position

                    2. Activity Analysis:
                        - Purple markers: High activity regions (>20 events)
                        - Red markers: Medium activity regions (10-20 events)
                        - Blue markers: Low activity regions (<10 events)
                        Marker size indicates intensity of recent activity

                    Active Screen Regions:
                    {chr(10).join([
                        f"- Region {region} ({data.get('screen_pos', 'unknown')}): "
                        f"{data.get('count', 0)} events, "
                        f"intensity {data.get('intensity', 0):.2f}, "
                        f"duration {data.get('duration', 0):.1f}s, "
                        f"pattern: {data.get('pattern_type', 'General Activity')}"
                        for region, data in understanding.get('active_regions', {}).items()
                    ])}

                    Recent Activities:
                    {chr(10).join([
                        f"- {act.get('type', 'motion')} at ({act.get('region', 'unknown')}) - "
                        f"Intensity: {act.get('intensity', 0):.2f}, "
                        f"Time: {time.strftime('%H:%M:%S', time.localtime(act.get('time', time.time())))}"
                        for act in understanding.get('recent_activities', [])[-5:]
                    ])}

                    Previous Context: {self.current_context if self.current_context else 'Starting analysis'}

                    Based on the screen content and activity patterns, provide a comprehensive analysis that could be used for automated interaction."""
                    ,
                    "images": [image_data]
                }
            else:
                prompt = {
                    "model": self.llm_model,
                    "prompt": f"""Analyze this screen activity data:
Screen Size: {screen_width}x{screen_height}

1. Visual Layout:
    - Describe the exact position and content of each window/application
    - Note any UI elements like buttons, menus, or interactive areas
    - Identify text content and its location
    - Describe any media content (images, videos) and their position

2. Activity Analysis:
    - Purple markers: High activity regions (>20 events)
    - Red markers: Medium activity regions (10-20 events)
    - Blue markers: Low activity regions (<10 events)
    Marker size indicates intensity of recent activity

Active Screen Regions:
{chr(10).join([
    f"- Region {region} ({data.get('screen_pos', 'unknown')}): "
    f"{data.get('count', 0)} events, "
    f"intensity {data.get('intensity', 0):.2f}, "
    f"duration {data.get('duration', 0):.1f}s, "
    f"pattern: {data.get('pattern_type', 'General Activity')}"
    for region, data in understanding.get('active_regions', {}).items()
])}

Recent Activities:
{chr(10).join([
    f"- {act.get('type', 'motion')} at ({act.get('region', 'unknown')}) - "
    f"Intensity: {act.get('intensity', 0):.2f}, "
    f"Time: {time.strftime('%H:%M:%S', time.localtime(act.get('time', time.time())))}"
    for act in understanding.get('recent_activities', [])[-5:]
])}

Previous Context: {self.current_context if self.current_context else 'Starting analysis'}

Based on the screen content and activity patterns, provide a comprehensive analysis that could be used for automated interaction."""
                }
            
            return prompt
            
        except Exception as e:
            print(f"[ERROR] Failed to build prompt: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return None

    def _send_llm_request(self, prompt, request_time):
        """Send request to remote inference server"""
        try:
            print("[DEBUG] Starting LLM request...")
            start_time = time.time()
            
            INFERENCE_SERVER = "https://z8ezes3rip4j66-8000.proxy.runpod.net/analyze"
            
            try:
                response = requests.post(
                    INFERENCE_SERVER,
                    json=prompt,
                    timeout=300,
                    verify=False,
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                )
                print(f"[DEBUG] Response received in {time.time() - start_time:.1f}s")
                
                if response.status_code == 200:
                    result = response.json()
                    self.current_context = result['response']
                    self.last_context_update = request_time
                    print(f"[DEBUG] Context updated: {self.current_context[:100]}...")
                else:
                    print(f"[ERROR] LLM request failed: {response.status_code}")
                    print(f"[ERROR] Response content: {response.text}")
                    
            except requests.exceptions.Timeout:
                print("[ERROR] LLM request timed out")
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request failed: {e}")
                
        except Exception as e:
            print(f"[ERROR] LLM request error: {e}")
        finally:
            self.pending_llm_request = False

    def _cleanup_resources(self):
        """Periodic cleanup"""
        current_time = time.time()
        
        # Cleanup old regions
        cutoff_time = current_time - self.context_update_interval
        self.active_regions = {
            k: v for k, v in self.active_regions.items()
            if v['last_seen'] > cutoff_time
        }
        
        # Clear old frames
        if hasattr(self, 'debug_dir') and os.path.exists(self.debug_dir):
            for f in os.listdir(self.debug_dir):
                if f.endswith('.png'):
                    try:
                        os.remove(os.path.join(self.debug_dir, f))
                    except:
                        pass

    def get_current_understanding(self) -> dict:
        """Get current screen understanding with frame"""
        try:
            process = psutil.Process()
            print(f"[DEBUG] Memory usage: {process.memory_percent()}%")
            
            current_time = time.time()
            
            # Generate visualization directly
            frame_base64 = self._generate_activity_visualization()
            
            # Format activities with proper structure
            recent_activities = []
            for act in list(self.activity_summary)[-5:]:
                if isinstance(act, dict) and all(k in act for k in ['type', 'region', 'screen_pos', 'intensity', 'time']):
                    recent_activities.append(act)
            
            understanding = {
                'screen_resolution': f"{self.reader.screen_width}x{self.reader.screen_height}",
                'active_regions': {
                    region: {
                        'count': data.get('count', 0),
                        'last_intensity': data.get('last_intensity', 0.0),
                        'first_seen': data.get('first_seen', current_time),
                        'last_seen': data.get('last_seen', current_time),
                        'screen_pos': data.get('screen_pos', (0, 0))
                    }
                    for region, data in self.active_regions.items()
                },
                'recent_activities': recent_activities,
                'current_context': self.current_context,
                'current_frame': frame_base64,
                'timestamp': current_time
            }
            
            return understanding
                
        except Exception as e:
            print(f"[ERROR] Failed to get understanding: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return {
                'error': str(e),
                'timestamp': time.time(),
                'current_frame': None,
                'current_context': '',
                'active_regions': {},
                'recent_activities': []
            }

    def stop(self):
        """Clean up resources on stop"""
        self.running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        if hasattr(self, 'reader'):
            self.reader.stop()
        # Clear buffers
        self.event_buffer.clear()
        self.activity_buffer.clear()
        self.context_buffer.clear()
        self.activity_summary.clear()
        self.active_regions.clear()

def main():
    analyzer = None
    try:
        analyzer = VisualActivityAnalyzer()
        print("[INFO] Visual activity analyzer running.")
        print("[INFO] Commands:")
        print("  - Press 'c' to print current context")
        print("  - Press 'u' to force context update")
        print("  - Press 'h' to show context history")
        print("  - Press 'd' to toggle debug mode")
        print("  - Press 'a' to perform automated action")  # New command
        print("  - Press 'q' to quit")
        
        while True:
            cmd = input().strip().lower()
            if cmd == 'a':
                goal = input("Enter goal (e.g., 'Click the search box'): ")
                context = analyzer.get_current_understanding()
                commands = analyzer.cursor_commander.generate_commands(context, goal)
                
                print("\nGenerated Commands:")
                for cmd in commands:
                    print(f"{cmd['type']} {' '.join(cmd['parameters'])} // {cmd['purpose']}")
                    
                if input("Execute these commands? (y/n): ").lower() == 'y':
                    for cmd in commands:
                        if not analyzer.cursor_commander.execute_command(cmd):
                            print(f"[ERROR] Failed to execute command: {cmd}")
                            break
                        time.sleep(0.1)  # Small delay between commands
            if cmd == 'c':
                understanding = analyzer.get_current_understanding()
                print("\n=== Current Context ===")
                print(f"Context: {understanding.get('current_context', 'No context yet')}")
                print("\nRecent Activities:")
                if understanding.get('recent_activities'):
                    for activity in understanding['recent_activities']:
                        print(f"- {activity['type']} at region {activity['region']} "
                            f"({activity['screen_pos']}) - "
                            f"Intensity: {activity['intensity']:.2f} - "
                            f"Time: {time.strftime('%H:%M:%S', time.localtime(activity['time']))}")
                else:
                    print("No activities recorded yet")
                print("\nActive Regions:")
                for region, data in understanding.get('active_regions', {}).items():
                    print(f"- Region {region}: {data['count']} events, "
                        f"Last intensity: {data['last_intensity']:.2f}")
                print("=====================\n")
            elif cmd == 'u':
                analyzer._update_context()
            elif cmd == 'h':
                print("\n=== Context History ===")
                if analyzer.context_buffer:
                    for ctx in analyzer.context_buffer:
                        print(f"\n[{time.strftime('%H:%M:%S', time.localtime(ctx['time']))}]")
                        print(ctx['context'])
                else:
                    print("No context history yet")
                print("=====================\n")
            elif cmd == 'd':
                analyzer.debug_mode = not analyzer.debug_mode
                print(f"Debug mode: {'enabled' if analyzer.debug_mode else 'disabled'}")
            elif cmd == 'q':
                break
            
    except KeyboardInterrupt:
        print("\n[INFO] Stopping analyzer...")
    except Exception as e:
        print(f"[ERROR] Failed to initialize analyzer: {e}")
    finally:
        if analyzer:  # Only try to stop if analyzer was created
            analyzer.stop()

if __name__ == "__main__":
    main()