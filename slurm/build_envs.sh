ENVNAME="glue"

PLATAFORM="a100" # Change to "a100" or "v100" as needed
PYTHON_VERSION="3.11"

COMPOSE_ENVNAME="$ENVNAME-$PLATAFORM"

module purge

module load arch/a100
module load cuda/12.6.3
module load miniforge/24.11.3
conda deactivate

# check if conda environment exists
if conda env list | grep -q "$COMPOSE_ENVNAME"; then
    conda env remove -n "$COMPOSE_ENVNAME" -y
fi

echo "Creating and activating new conda environment: $COMPOSE_ENVNAME"
conda create -n "$COMPOSE_ENVNAME" python=$PYTHON_VERSION -y
conda activate "$COMPOSE_ENVNAME"
pip install -e .
# pip install -U xformers --index-url https://download.pytorch.org/whl/cu126


# PLATAFORM="h100" # Change to "a100" or "v100" as needed
# PYTHON_VERSION="3.11"

# COMPOSE_ENVNAME="$ENVNAME-$PLATAFORM"

# module purge

# module load arch/h100
# module load cuda/12.6.3
# module load miniforge/24.11.3
# conda deactivate

# # check if conda environment exists
# if conda env list | grep -q "$COMPOSE_ENVNAME"; then
#     conda env remove -n "$COMPOSE_ENVNAME" -y
# fi
# echo "Creating and activating new conda environment: $COMPOSE_ENVNAME"
# conda create -n "$COMPOSE_ENVNAME" python=$PYTHON_VERSION -y
# conda activate "$COMPOSE_ENVNAME"
# pip install --no-cache-dir -e .
# # pip install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu126