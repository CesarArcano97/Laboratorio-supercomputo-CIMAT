# Uso del Laboratorio de Supercómputo CIMAT

## Acceso y prueba inicial en el Laboratorio de Supercómputo del Bajío (CIMAT)

Este documento registra los pasos realizados para conectarnos y validar el uso de GPUs (NVIDIA TITAN RTX) en el cluster **Lab-SB** de CIMAT. Servirá como guía de referencia para futuros proyectos.

---

## Acceso vía SSH (desde Linux)

1. Conectarse al **front-end** del cluster:
   ```bash
   ssh -p <port> <username>@<host>

* \<port\> → puerto asignado en correo de alta.

* \<username\> → tu usuario (ejemplo: est_posgrado_cesar.aguirre).

* \<host\> → dirección del cluster (ejemplo: el-insurgente.cimat.mx).

### Verificamos el directorio home:

```bash
pwd
ls
```

## Revisión de particiones disponibles de GPU

Al consultar ```sinfo``` se identificaron las particiones con GPUs:

```bash
sinfo -o "%P %G %N" | grep GPU
```

Salida:

```scss
GPU (null) g-0-[1-12]
GPUX (null) g-0-[1-12]
GPU_PROY2025 (null) g-0-[4-9]
GPU_WS (null) g-ws-[1-3] 
```

Esto confirma que los nodos ```g-0-x``` tienen 2x NVIDIA TITAN RTX (24 GB) cada uno.


## Prueba inicial con Slurm 

Se crea un archivo de ```gpu_test.slurm```:

```bash
#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=GPU
#SBATCH --time=00:05:00
#SBATCH --output=gpu_test_%j.out

# Validar GPU disponible
nvidia-smi
```

Ejecutar:

```bash
sbatch gpu_test.slurm
squeue -u $USER
```

Revisar el log::

```bash
cat gpu_test_<jobid>.out
```

El resultado debe ser algo del estilo:

```lua
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08   Driver Version: 545.23.08   CUDA Version: 12.3                 |
| GPU  Name          Persistence-M | Bus-Id | Temp | Memory-Usage | GPU-Util | Compute M|
|  0  NVIDIA TITAN RTX            ...       | 24 GB |              |         |          |
|  1  NVIDIA TITAN RTX            ...       | 24 GB |              |         |          |
+---------------------------------------------------------------------------------------+
| No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

Esto confirma el acceso exitoso a los nodos de la GPU, y su disponibilidad. Además de la versión de CUDA preparada para usarse. 


## Instalación de Miniconda en $HOME

```bash
cd $HOME
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Inicialización de conda:

```bash
conda init bash
source ~/.bashrc
```

## Entorno Conda para Deep Learning

Creación del entorno ```prometheus```:

```bash
conda create -n prometheus python=3.11 -y
conda activate prometheus
```

Instalación de PyTorch con soporte GPU (CUDA 12.1):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

## Prueba de PyTorch + GPU en Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=torch_gpu_test
#SBATCH --partition=GPU
#SBATCH --time=00:05:00
#SBATCH --output=torch_gpu_test_%j.out

# Inicializar conda en nodos Slurm
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate prometheus

python - << 'EOF'
import torch
print("PyTorch versión:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU detectada:", torch.cuda.get_device_name(0))
    print("Versión CUDA en PyTorch:", torch.version.cuda)
EOF
```

Ejecutar:

```bash
sbatch torch_gpu_test.slurm
```

Salida esperada:

```bash
PyTorch versión: 2.5.1
CUDA disponible: True
GPU detectada: NVIDIA TITAN RTX
Versión CUDA en PyTorch: 12.1
```
