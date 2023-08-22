#! /bin/bash
set -eux

# Download heart disease dataset following https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_heart_disease/README.md

package_name="flamby"
package_path=$(python -c "import os; import $package_name; print(os.path.dirname($package_name.__file__))")

if [ -n "$package_path" ]; then
    echo "The path of '$package_name' package is: $package_path"
else
    echo "'$package_name' package not found"
fi

cd "$package_path/datasets/fed_heart_disease/dataset_creation_scripts" && \
python download.py --output-folder "$package_path/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset"

echo "OK."
