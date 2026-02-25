## Tested System Configuration

This project was tested on the following system:

- **Operating System:** Ubuntu 22.04 LTS  
- **CPU:** [Your CPU model]  
- **RAM:** [e.g., 32 GB]  
- **GPU (if used):** [GPU model + VRAM]  
- **CUDA Version (if applicable):** [e.g., 12.2]  
- **NVIDIA Driver Version (if applicable):** [driver version]  
- **Python Version:** 3.10  
- **Conda Version:** [conda version]  
- **Ollama Version:** [ollama version]  
- **LLM Model Used:** gemma3:12b  
- **Whisper Backend:** [faster-whisper / openai-whisper]  
- **Whisper Model Size:** [tiny / base / small / etc.]  

The code may work on other systems, but this is the configuration under which it was verified.

## System Dependencies (Ubuntu 22.04)

Some audio components require SoX.

Install SoX using:

```bash
sudo apt update
sudo apt install sox libsox-fmt-all -y
```

Verify installation:

```bash
sox --version
```

If the version prints correctly, SoX is installed.

# Setup Instructions 

## 1. Install Miniconda (Linux: Ubuntu 22.04)

### Step 1: Download Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Run the installer

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

During installation:

- Press `Enter` to scroll through the license  
- Type `yes` to accept the license  
- Press `Enter` to confirm the default install location  
- Type `yes` when asked to initialize Miniconda  

### Step 3: Reload your shell

```bash
source ~/.bashrc
```

### Step 4: Verify installation

```bash
conda --version
```

Expected output:

```bash
conda 24.x.x
```

## 2. Create Conda Environment

Create a new environment for the project:

```bash
conda create -n voice_chat python=3.10 -y
```

Activate the environment:

```bash
conda activate voice_chat
```

Verify that the environment is active:

```bash
which python
```

It should point to a path inside:

```
~/miniconda3/envs/voice_chat/
```

## 3. Navigate to Project Folder

Move into the project directory:

```bash
cd llm_powered_social_robot
```

Verify you are in the correct folder:

```bash
ls
```

You should see files such as:

```
voice_chat.py
README.md
...
```

## 4. Install NumPy

Install NumPy inside the active `voice_chat` environment:

```bash
pip install numpy faster-whisper scipy torch qwen-tts 
```

Verify installation:

```bash
python -c "import numpy; print(numpy.__version__)"
```

## 5. Install Ollama (Ubuntu 22.04)

Install Ollama using the official installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installation, verify:

```bash
ollama --version
```

Start Ollama service (if not auto-started):

```bash
ollama serve
```

In a new terminal, pull the required model:

```bash
ollama pull gemma3:12b
```

Test the model:

```bash
ollama run gemma3:12b
```