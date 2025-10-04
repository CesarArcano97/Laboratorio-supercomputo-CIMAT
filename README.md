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

### Descarga de datos cluster a máquina externa:

## Descargar resultados a mi PC (fuera de CIMAT)

Cuando se está fuera de CIMAT, se debe usar el **host** indicado y **puerto** indicado en el registro.

#### Opción rápida (`scp`)
> Nota: la `-P` de puerto es **mayúscula**.
```bash
scp -P <PUERTO> -r \
  <USUARIO>@<HOST>:~/DIRECTORIO-DE-INTERES \
  /home/USUARIO/
```
# Fine-tuning y Generación Offline con Mistral-7B + Unsloth (LoRA)

Este flujo documenta los pasos funcionales para entrenar y generar letras de canciones **offline** en el servidor **el-insurgente (CIMAT)** utilizando **Mistral-7B-Instruct** con **LoRA eficiente (Unsloth)**.

---

## 1. Estructura del proyecto

```
mistral-project/
│
├── data/
│   └── processed/
│       └── lyrics_train.jsonl          ← corpus a nivel texto
│
├── models/
│   ├── base/
│   │   └── Mistral-7B-Instruct-v0.2    ← modelo base descargado
│   └── finetuned/
│       └── t21p_lr2e-4_ep3_bs2x4_YYYYMMDD_HHMMSS/
│           └── final_model/            ← adaptadores LoRA + tokenizer
│
├── results/
│   └── generations/                    ← letras generadas
│
├── logs/
│   └── train_.out, generate_.out
│
└── src/
    ├── train/train_mistral_offline.py
    └── generate/generate_mistral_offline.py
```

---

## 2. Entorno Conda funcional

```bash
conda create -n mistral-env python=3.10 -y
conda activate mistral-env

pip install torch==2.8.0 \
            transformers==4.51.3 \
            accelerate==1.1.1 \
            trl==0.9.4 \
            peft==0.11.1 \
            unsloth==2025.9.7 \
            bitsandbytes==0.43.1 \
            xformers==0.0.32.post2 \
            datasets==2.21.0
```
**Versión estable confirmada:**
- **Torch 2.8.0** + **CUDA 12.8**
- **Unsloth 2025.9.7** compatible con Mistral 7B.

---

## 3. Entrenamiento offline con LoRA

**Archivo:** `src/train/train_mistral_offline.py`

**Puntos clave:**

- **Modo totalmente offline** configurado con:
  ```python
  import os
  os.environ["HF_HUB_OFFLINE"] = "1"
  os.environ["TRANSFORMERS_OFFLINE"] = "1"
  os.environ["HF_DATASETS_OFFLINE"] = "1"
  os.environ["UNSLOTH_FORCE_OFFLINE"] = "1"
  ```
- **Carga local** desde: `~/mistral-project/models/base/Mistral-7B-Instruct-v0.2`
- **LoRA** aplicada sólo en proyecciones Q/K/V/O (≈ 1 % de parámetros entrenados).
- **Gradiente acumulado** para lotes pequeños (`grad_accum=4`).
- **Guardado automático** de adaptadores + tokenizer en: `models/finetuned/<run_name>/final_model/`

**Ejemplo de comando Slurm:**
```bash
#!/bin/bash
#SBATCH --job-name=train_mistral
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral-env
cd ~/mistral-project/src/train

python train_mistral_offline.py
```

**Resultado exitoso:**
```bash
Trainable parameters = 13.6M (0.19% of total)
train_loss ≈ 1.59
Modelo guardado en: models/finetuned/.../final_model/
```

---

## 🎤 4. Generación offline

**Archivo:** `src/generate/generate_mistral_offline.py`

- Combina modelo base + adaptadores LoRA vía `PeftModel.from_pretrained`.
- Usa `device_map="auto"` (carga en GPU automáticamente).
- **Prompts configurables:**
  ```python
  prompts = [
      "In the city lights I find myself",
      "They say I'm broken but I'm breathing",
      "Sometimes my shadow sings louder than I do"
  ]
  ```
- **Parámetros de sampling:**
  ```python
  max_new_tokens=220, temperature=0.9, top_p=0.95, do_sample=True
  ```
- Letras generadas en: `results/generations/song_1.txt`, `song_2.txt`, etc.

**Ejemplo de Slurm de inferencia:**
```bash
#!/bin/bash
#SBATCH --job-name=generate_mistral
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral-env
cd ~/mistral-project/src/generate

python generate_mistral_offline.py
```
**Salida esperada:**
```bash
🔹 Cargando modelo base en GPU...
🔹 Cargando adaptadores LoRA...
Generando canción 1...
Guardada: results/generations/song_1.txt
...
```

---

## 5. Ejemplo de letra generada

**Prompt:** “In the city lights I find myself”

```
In the city lights I find myself
I don't know why
I wanna stay inside tonight
I think it's right
But my heart keeps telling me that I should go...
```
Original, coherente y estilísticamente alineada con Twenty One Pilots.
Sin coincidencias literales con letras oficiales (verificado manualmente).

---

## 6. Notas finales

- **Unsloth con LoRA** ofrece ≈ **2× menor VRAM** y entrenamiento estable en 1 GPU TITAN RTX (24 GB).
- Todos los procesos se ejecutaron **offline**, sin conexión a Hugging Face.
- La ruta más reciente del modelo entrenado:
  ```bash
  ~/mistral-project/models/finetuned/t21p_lr2e-4_ep3_bs2x4_20251004_033651/final_model
  ```

**Estado actual del pipeline:** Fine-tuning completo + generación funcional offline.
