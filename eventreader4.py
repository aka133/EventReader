import objc
from Foundation import NSObject, NSRunLoop, NSThread, NSDate, NSDefaultRunLoopMode
from AppKit import NSScreen
from CoreMedia import CMTimeMake, CMSampleBufferGetImageBuffer
import numpy as np
import time
from PIL import Image
import io
from ScreenCaptureKit import (
    SCStream, SCStreamConfiguration, SCShareableContent, 
    SCContentFilter, SCStreamOutputTypeScreen
)
from Quartz import (
    CVPixelBufferLockBaseAddress, CVPixelBufferUnlockBaseAddress,
    CVPixelBufferGetBaseAddress, CVPixelBufferGetWidth,
    CVPixelBufferGetHeight, kCVPixelFormatType_32BGRA,
    CVPixelBufferGetBytesPerRow
)
import cv2
from collections import deque
import ctypes
import threading

SCStreamDelegate = objc.protocolNamed('SCStreamDelegate')

class ScreenEventDelegate(NSObject):
    __pyobjc_protocols__ = [SCStreamDelegate]

    def init(self):
        self = objc.super(ScreenEventDelegate, self).init()
        if self is None: 
            return None
            
        self.last_event_time = 0
        self.min_event_interval = 1.0 / 30.0  # 30 fps max
        
        self.debug_mode = True
        self.frame_skip = 1
        self.min_threshold = 0.1
        self.previous_buffer = None
        self.callback = None
        self._frame_counter = 0
        self.current_frame = None
        self._frame_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self.frame_buffer = {}  # Store frames by region
        self.frame_count = 0
        self.last_debug_time = time.time()
        return self

    def _get_reduced_frame(self, pixel_buffer):
        """Convert pixel buffer to numpy array and reduce resolution"""
        try:
            width = CVPixelBufferGetWidth(pixel_buffer)
            height = CVPixelBufferGetHeight(pixel_buffer)
            bytesPerRow = CVPixelBufferGetBytesPerRow(pixel_buffer)
            
            CVPixelBufferLockBaseAddress(pixel_buffer, 0)
            try:
                base_address = CVPixelBufferGetBaseAddress(pixel_buffer)
                if base_address is None:
                    print("[ERROR] Failed to get base address")
                    return None
                    
                # CRITICAL FIX: Use correct buffer size
                buffer_size = height * bytesPerRow
                buffer_data = base_address.as_buffer(buffer_size)
                
                # Create frame with correct shape
                frame = np.frombuffer(buffer_data, dtype=np.uint8)
                frame = frame.reshape((height, bytesPerRow // 4, 4))  # Note the bytesPerRow division
                frame = frame[:, :width]  # Trim any padding
                
                # Convert BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Reduce resolution
                small_frame = cv2.resize(frame, (width // 4, height // 4))
                
                if self.debug_mode:
                    print(f"[DEBUG] Raw frame stats:")
                    print(f"  Shape before resize: {frame.shape}")
                    print(f"  Shape after resize: {small_frame.shape}")
                    print(f"  Range: {small_frame.min()}-{small_frame.max()}")
                    print(f"  Mean: {small_frame.mean()}")
                    
                return small_frame
                
            finally:
                CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
                
        except Exception as e:
            print(f"[ERROR] Frame reduction failed: {e}")
            traceback.print_exc()
            return None

    def setCallback_(self, callback):
        """Set the event callback function"""
        self.callback = callback
        print(f"[DEBUG] Callback set: {callback is not None}")

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, type_):
        try:
            current_time = time.time()
            if current_time - self.last_event_time < self.min_event_interval:
                return
            
            pixel_buffer = CMSampleBufferGetImageBuffer(sample_buffer)
            if pixel_buffer is None:
                return
            
            CVPixelBufferLockBaseAddress(pixel_buffer, 0)
            try:
                # Get reduced frame more efficiently
                grayscale = self._get_reduced_frame(pixel_buffer)
                if grayscale is None:
                    return
                    
                if self.previous_buffer is not None:
                    # Compute delta more efficiently
                    delta = np.abs(grayscale - self.previous_buffer)
                    changes = np.where(delta > self.min_threshold)
                    
                    if len(changes[0]) > 0:
                        # Process changes in batches
                        for y, x in zip(changes[0], changes[1]):
                            intensity = delta[y, x]
                            if intensity > self.min_threshold and self.callback:
                                full_x = int(x * 4)
                                full_y = int(y * 4)
                                self.callback(full_x, full_y, float(intensity))
                                
                self.previous_buffer = grayscale.copy()
                self.last_event_time = current_time
                
            finally:
                CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
                
        except Exception as e:
            print(f"[ERROR] Frame processing error: {e}")
            traceback.print_exc()

    def _process_region(self, base_address, bytes_per_row, x1, y1, x2, y2):
        """Efficient region processing"""
        try:
            # Process region in smaller chunks
            chunk_size = 32  # Process in 32-pixel chunks
            total_intensity = 0
            samples = 0
            
            for y in range(y1, y2, chunk_size):
                chunk_y2 = min(y + chunk_size, y2)
                for x in range(x1, x2, chunk_size):
                    chunk_x2 = min(x + chunk_size, x2)
                    
                    # Sample center pixel of chunk
                    center_y = (y + chunk_y2) // 2
                    center_x = (x + chunk_x2) // 2
                    offset = center_y * bytes_per_row + center_x * 4
                    pixel_data = base_address[offset:offset+3]  # RGB only
                    
                    total_intensity += sum(pixel_data) / (3 * 255)
                    samples += 1
            
            return total_intensity / samples if samples > 0 else 0
            
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Region processing error: {e}")
            return 0.0
        
    def _cleanup_regions(self, current_time):
        """Remove inactive regions"""
        cleanup_threshold = current_time - 1.0  # 1 second timeout
        self.active_regions = {
            k: v for k, v in self.active_regions.items()
            if v['last_seen'] > cleanup_threshold
        }

    def _apply_temporal_filter(self, region_key, intensity):
        """Filter changes based on their temporal patterns"""
        try:
            current_time = time.time()
            
            if region_key not in self.temporal_buffer:
                self.temporal_buffer[region_key] = deque(maxlen=10)
                
            buffer = self.temporal_buffer[region_key]
            buffer.append((current_time, intensity))
            
            # Analyze temporal pattern
            if len(buffer) >= 3:
                recent_intensities = [i for _, i in buffer]
                avg_intensity = sum(recent_intensities) / len(recent_intensities)
                variance = sum((i - avg_intensity) ** 2 for i in recent_intensities) / len(recent_intensities)
                
                # Classify the activity pattern
                if variance > 0.1:  # High variance indicates sporadic changes (possible noise)
                    return None
                elif avg_intensity > 0.3:  # Sustained high intensity indicates real activity
                    return {
                        'type': 'sustained',
                        'intensity': avg_intensity,
                        'duration': current_time - buffer[0][0]
                    }
                else:  # Low intensity changes might be background activity
                    return {
                        'type': 'background',
                        'intensity': avg_intensity,
                        'duration': current_time - buffer[0][0]
                    }
                
        except Exception as e:
            print(f"[ERROR] Temporal filtering error: {e}")
            return None

    def _cluster_changes(self, changes, delta_map):
        """Group nearby changes into coherent regions"""
        try:
            regions = []
            visited = set()
            
            for i in range(len(changes[0])):
                y, x = changes[0][i], changes[1][i]
                if (x,y) in visited:
                    continue
                    
                # Start a new region
                region = {(x,y)}
                queue = [(x,y)]
                min_x, min_y = x, y
                max_x, max_y = x, y
                
                # Grow region to include nearby changes
                while queue:
                    cx, cy = queue.pop(0)
                    # Check 8 neighboring pixels
                    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                        nx, ny = cx + dx, cy + dy
                        if (nx,ny) not in visited and delta_map[ny,nx] > self.min_threshold:
                            queue.append((nx,ny))
                            region.add((nx,ny))
                            visited.add((nx,ny))
                            min_x = min(min_x, nx)
                            max_x = max(max_x, nx)
                            min_y = min(min_y, ny)
                            max_y = max(max_y, ny)
                
                # Only keep regions with significant activity
                if len(region) > 5:  # Minimum size threshold
                    regions.append((min_x, min_y, max_x, max_y))
                    
            return regions
            
        except Exception as e:
            print(f"[ERROR] Clustering error: {e}")
            return []

    def get_current_frame(self):
        with self._frame_lock:
            if self.current_frame is None:
                return np.zeros((
                    self.screen_height // 4,
                    self.screen_width // 4,
                    3
                ), dtype=np.uint8), time.time()
            return np.ascontiguousarray(self.current_frame).copy(), time.time()

    def _recover_from_error(self):
        """Attempt to recover from stream errors"""
        try:
            with self._frame_lock:
                self.current_frame = None
                self.previous_buffer = None
            
            # Restart stream
            self.stop()
            time.sleep(1)  # Wait before restart
            self.__init__(self.callback)  # Reinitialize
            
        except Exception as e:
            print(f"[ERROR] Recovery failed: {e}")

class EventDrivenScreenReader:
    def __init__(self, event_callback=None):
        # Core attributes
        self.debug_mode = True
        self.event_callback = event_callback  # Store callback as instance variable
        self.delegate = ScreenEventDelegate.alloc().init()
        self.delegate.setCallback_(event_callback)
        self.stream = None
        self._strong_refs = []
        
        # Screen dimensions
        screen = NSScreen.mainScreen()
        self.screen_width = int(screen.frame().size.width)
        self.screen_height = int(screen.frame().size.height)
        
        # Initialize capture
        self.setup_capture_session()

    def get_current_frame(self):
        """Get the current frame from the delegate"""
        if not hasattr(self.delegate, 'current_frame') or self.delegate.current_frame is None:
            return np.zeros((
                self.screen_height // 4,
                self.screen_width // 4,
                3
            ), dtype=np.uint8), time.time()
        return self.delegate.current_frame.copy(), time.time()

    def setup_capture_session(self):
        # Add configuration settings
        self.frame_buffer_size = 2  # Keep only 2 frames in memory
        self.min_frame_interval = 1.0 / 30.0  # 30 fps max
        self.processing_enabled = True
        
        # Initialize locks
        self._frame_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self.current_frame = None
        self.previous_frame = None
        self.event_buffer = deque(maxlen=100)  # Replace temporal_buffer
        self.active_regions = {}
        
        self.processing_levels = {
            'idle': {'fps': 5, 'resolution': 8},  # 1/8th resolution
            'active': {'fps': 15, 'resolution': 4},  # 1/4th resolution
            'focused': {'fps': 30, 'resolution': 2}  # 1/2th resolution
        }
        self.current_level = 'idle'
        
        def handle_content(shareable_content, error):
            if error:
                print(f"[ERROR] Failed to get shareable content: {error}")
                return
            
            try:
                displays = shareable_content.displays()
                if not displays:
                    print("[ERROR] No displays found")
                    return
                
                display = displays[0]
                self.screen_width = int(display.frame().size.width)
                self.screen_height = int(display.frame().size.height)
                
                if self.debug_mode:
                    print(f"[DEBUG] Display info:")
                    print(f"  - Frame: {self.screen_width}x{self.screen_height}")
                    print(f"  - Display ID: {display.displayID()}")
                
                # Configure stream
                config = SCStreamConfiguration.alloc().init()
                config.setWidth_(self.screen_width)
                config.setHeight_(self.screen_height)
                config.setQueueDepth_(1)
                config.setShowsCursor_(True)
                config.setMinimumFrameInterval_(CMTimeMake(1, 30))
                config.setPixelFormat_(kCVPixelFormatType_32BGRA)
                
                # Create filter
                filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                    display, None)
                
                # Initialize delegate
                self.delegate = ScreenEventDelegate.alloc().init()
                self.delegate.setCallback_(self.event_callback)
                
                # Create stream
                self.stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                    filter, config, self.delegate)
                
                # Add output
                error_ptr = objc.nil
                self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    self.delegate,
                    SCStreamOutputTypeScreen,
                    None,  # Use default queue
                    error_ptr
                )
                
                # Start capture
                self.stream.startCaptureWithCompletionHandler_(
                    lambda error: print("[DEBUG] Stream started" if not error else f"[ERROR] {error}")
                )
                
            except Exception as e:
                print(f"[ERROR] Setup failed: {e}")
                traceback.print_exc()

        print("[DEBUG] Getting shareable content...")
        SCShareableContent.getShareableContentWithCompletionHandler_(handle_content)

    def stop(self):
        try:
            if self.stream:
                self.stream.stopCaptureWithCompletionHandler_(lambda error: 
                    print("Stream stopped" if not error else f"Stop failed: {error}"))
            
            # Clear frame buffers
            with self._frame_lock:
                self.current_frame = None
                self.previous_frame = None
            
            # Clear other buffers
            self.event_buffer.clear()
            self.active_regions.clear()
            
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {e}")
            traceback.print_exc()

    def _adjust_processing_level(self, activity_count):
        if activity_count > 10:
            self.current_level = 'focused'
        elif activity_count > 5:
            self.current_level = 'active'
        else:
            self.current_level = 'idle'

def main():
    def custom_event_handler(x: int, y: int, intensity: float):
        print(f"[EVENT] Motion at ({x}, {y}) - Intensity: {intensity:.3f}")
    
    reader = EventDrivenScreenReader(event_callback=custom_event_handler)
    
    try:
        runLoop = NSRunLoop.currentRunLoop()
        while True:
            runLoop.runMode_beforeDate_(
                "kCFRunLoopDefaultMode",
                NSDate.dateWithTimeIntervalSinceNow_(0.1)  # Reduced polling frequency
            )
    except KeyboardInterrupt:
        print("\nStopping...")
        reader.stop()

if __name__ == "__main__":
    main()
    main()