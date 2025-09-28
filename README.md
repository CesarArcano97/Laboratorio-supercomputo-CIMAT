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

<h2> Ejecución exitosa en el clúster Lab-SB (CIMAT)</h2>

<p>Se logró entrenar y ejecutar un modelo <code>tiny-gpt2</code> en el laboratorio de supercómputo de CIMAT, siguiendo buenas prácticas de organización y uso de SLURM.</p>

<h3> Estructura del proyecto</h3>

<pre>
proyecto_gpt2/
├── data/               # Corpus de entrenamiento
│   └── TOP_corpus_generativo_unificado.txt
├── src/                # Código fuente
│   └── train.py
├── models/             # Modelos locales y checkpoints
│   └── tiny-gpt2/      # Modelo predescargado desde Hugging Face
├── logs/               # Logs de SLURM
│   └── gpt2_finetune-<JOBID>.log
├── results/            # Métricas y outputs
└── run_entrenamiento.sh # Script de lanzamiento con SLURM
</pre>

<h3> Pasos realizados</h3>

<ol>
  <li>Crear un entorno <code>conda</code> específico para NLP/Transformers:
    <pre>conda create -n prometheus python=3.11 -y</pre>
  </li>
  <li>Instalar dependencias necesarias:
    <pre>conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets protobuf</pre>
  </li>
  <li>Descargar el modelo <code>tiny-gpt2</code> en local y subirlo al clúster (ya que los nodos no tienen internet).</li>
  <li>Colocar el corpus en <code>data/TOP_corpus_generativo_unificado.txt</code>.</li>
  <li>Lanzar entrenamiento de prueba en CPU desde la terminal:
    <pre>
python src/train.py \
  --model_name "./models/tiny-gpt2" \
  --train_file "./data/TOP_corpus_generativo_unificado.txt" \
  --epochs 2
    </pre>
  </li>
</ol>

<h3> Resultados</h3>

<ul>
  <li>Entrenamiento completado exitosamente (2 épocas, 100 steps).</li>
  <li>Pérdida final aproximada: <code>train_loss ≈ 10.74</code>.</li>
  <li>Modelo guardado en: <code>./models/final</code>.</li>
  <li>Generación de texto con el prompt inicial funcionando.</li>
</ul>

<h3> Cosas a tomar en cuenta</h3>

<ul>
  <li>Los nodos del clúster <b>no tienen internet</b>, por lo que los modelos/tokenizers deben descargarse previamente y transferirse completos.</li>
  <li>Es esencial mantener una <b>estructura de proyecto organizada</b> (data, src, models, logs, results).</li>
  <li>El uso de <code>run_entrenamiento.sh</code> con <code>sbatch</code> facilita reproducir experimentos en GPU.</li>
  <li>Los logs de SLURM (<code>logs/gpt2_finetune-*.log</code>) son la primera fuente para depuración.</li>
</ul>

_________________________________________________________

# Proyecto llama-beta

Este documento recopila los pasos que funcionaron en el **cluster Lab-SB de CIMAT** para afinar el modelo `TinyLLaMA` sobre un corpus propio (canciones de TOP).

---

## Estructura del proyecto

```arduino
llama-beta/
├── data/
│ └── TOP_corpus_generativo_unificado.txt
├── logs/ # logs de SLURM
├── models/
│ └── tiny-llama-1b/ # modelo predescargado desde laptop
├── scripts/
│ └── run_train.sh # script de entrenamiento con SLURM
└── src/
└── train.py # script de fine-tuning
```

### Entrenamiento en el cluster

Se lanza el entrenamiento con SLURM:

```bash
cd ~/llama-beta
sbatch scripts/run_train.sh
```

Revisar si el job sigue en cola o ya corriendo:

```bash
squeue -u $USER
```

Monitoreo y logs

Para ver el log en vivo:

```bash
tail -f logs/llama_beta_train-<JOBID>.log
```

Al terminar, puedes descargar el log a tu PC (corriendo desde el server):

```bash
ssh <USUARIO>@el-insurgente.cimat.mx "cat ~/llama-beta/logs/llama_beta_train-<JOBID>.log" > llama_beta_train-<JOBID>.log
```

### Puedes añadir notificaciones para ver cuando termine tu log

Agregando al script de train.sh:

```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tu_correo@cimat.mx
```

O bien un monitoreo manual:
```bash
#!/usr/bin/env bash
JOBID="$1"
[ -z "$JOBID" ] && { echo "Uso: $0 <JOBID>"; exit 1; }

echo "Esperando a que termine el job $JOBID..."
while squeue -j "$JOBID" -h >/dev/null 2>&1; do sleep 60; done

STATE=$(sacct -j "$JOBID" --format=State --noheader | head -n1 | awk '{print $1}')
[ -z "$STATE" ] && STATE="FINISHED"
echo -e "Job $JOBID terminó con estado: $STATE\a"
```

Uso: `scripts/notify_when_done.sh <JOBID>`
