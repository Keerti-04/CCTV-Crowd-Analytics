# CCTV Crowd Analytics

## Project Overview

CCTV Crowd Analytics is a surveillance video analysis application that estimates crowd density and identifies congested and free zones.

The application processes uploaded CCTV footage and displays:
- Estimated crowd count.
- Crowd density visualization using heatmaps.
- Zone classification: FREE, MODERATE, or CONGESTED.
- Dashboard showing crowd analysis results.

## Technologies Used

- Python
- Streamlit
- PyTorch
- CSRNet with SE Attention
- OpenCV
- NumPy
- Matplotlib
- Supabase
- Docker

## Project Structure

```text
CCTV-Crowd-Analytics/
│
├── app.py
├── estimation.py
├── csrnet_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── models/
│   └── CSRNet_SE.pth
│
└── .streamlit/
    └── secrets.toml  (created locally; not included in GitHub)
```

## Prerequisites

Install **Docker Desktop** (Windows) or Docker Engine (Linux).

Python and the project dependencies do not need to be installed separately because they are installed inside the Docker image.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Keerti-04/CCTV-Crowd-Analytics.git
cd CCTV-Crowd-Analytics
```

### 2. Configure Supabase

Create a `.streamlit` folder in the project directory.

Inside it, create a file named `secrets.toml`:

```toml
[supabase]
url = "YOUR_SUPABASE_PROJECT_URL"
key = "YOUR_SUPABASE_API_KEY"
```

Replace the placeholders with valid Supabase credentials.

The Supabase project must have the required `camera_configs` table and appropriate permissions.

**Important:** Never upload `secrets.toml` or expose your Supabase credentials publicly.

### 3. Build the Docker Image

Run:

```bash
docker build -t cctv-crowd-analytics .
```

Docker automatically installs the required Python dependencies and copies the application files into the image.

### 4. Run the Application

**Windows PowerShell:**

```powershell
docker run -d --name cctv-demo -p 8501:8501 -v "${PWD}/.streamlit:/app/.streamlit" cctv-crowd-analytics:latest
```

### 5. Access the Application

Open the following URL in your browser:

```text
http://localhost:8501
```

Upload a surveillance video and click **Start Crowd Analysis** to view the results.

## Managing the Docker Container

To stop the application:

```bash
docker stop cctv-demo
```

To start it again:

```bash
docker start cctv-demo
```

To check the running container:

```bash
docker ps
```

To view container logs:

```bash
docker logs cctv-demo
```

## Model Information

The project uses a CSRNet-based crowd counting model with Squeeze-and-Excitation (SE) attention.

- Training dataset: ShanghaiTech Part A.
- Model weights: `models/CSRNet_SE.pth`.
- Model inference is performed using PyTorch.

The training dataset is not included in this repository. The trained model weights are provided for inference.

## Notes

- Docker must be installed and running before launching the application.
- Valid Supabase credentials are required for database functionality.
- The application uses CPU-based inference.
- The Docker image includes the application and its Python dependencies.