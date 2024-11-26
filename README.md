# EventReader

## Project Overview
This repository documents an event-driven computer vision system designed for efficient computer interaction. Inspired by Anthropic's Computer Use research, the system implements an alternative approach to continuous screenshot analysis, focusing on event-based processing for reduced resource utilization.

## Project Evolution & Technical Components

### 1. Event Reader System (`eventreader4.py`)
My initial implementation focuses on efficient screen monitoring. The system implements native OS-level screen capture using ScreenCaptureKit for macOS and Win32API for Windows environments. I developed pixel-level delta detection using NumPy-based grayscale intensity analysis, utilizing memory-mapped arrays for efficient data handling. The system implements spatial-temporal filtering through scipy.signal processing for event detection.

### 2. Command Control Architecture (`cursorcommander.py`)
Building on the event detection system, I developed a cursor control system that translates high-level intentions into precise movements. The implementation features command parsing using regex-based pattern matching and coordinate validation through NumPy-based boundary checking. The system implements basic screen-space transformations and integrates with external LLM services through HTTP requests.

### 3. Neural Inference System (`inference_server.py`)
The inference system implements machine learning capabilities using PyTorch and modern vision-language models. Running on RTX 3090 hardware, I implemented integration with BakLLaVA-1 and Ollama models for visual understanding and command generation. The system features basic API endpoints for model inference and implements error handling for robust operation.

### 4. State Management & Integration (`integrator.py`)
I developed an event processing system that maintains temporal context using circular buffers. The implementation features activity tracking and basic rate limiting. The system manages state transitions and implements error recovery mechanisms for stable operation.

## Technical Achievements & Innovations

### Vision System Architecture
The system implements an efficient alternative to screenshot-based approaches through event-driven processing. By focusing on pixel-level changes and efficient memory management, the architecture provides a foundation for lightweight computer interaction systems.

### Model Integration
The system integrates with modern vision-language models, utilizing BakLLaVA's ViT-L/14 vision encoder for visual understanding. I implemented custom Ollama model serving with basic API communication and response handling, allowing for flexible model deployment.

### Processing Pipeline
The system implements input processing with pixel-level delta detection and basic thresholding. I developed event stream processing using temporal filtering and spatial analysis. The system features ROI detection for focused processing of relevant screen regions.

## Technologies & Dependencies
- PyTorch with CUDA support
- FastAPI for basic API endpoints
- Transformers library for model integration
- NumPy for numerical operations
- OpenCV for image processing
- Scikit-learn for basic analysis

## Future Development
The project roadmap includes:
- Implementation of OSWorld benchmark testing
- Development of LLM fine-tuning for cursor commands
- Enhancement of event processing algorithms
- Integration with additional vision-language models
- Implementation of comprehensive performance metrics

## Contact
archit.kalra@rice.edu
