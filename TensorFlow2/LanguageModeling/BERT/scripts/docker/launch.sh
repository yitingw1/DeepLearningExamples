#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

CMD=${@:-/bin/bash}
NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

docker run --gpus $NV_VISIBLE_DEVICES -itd \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name=bert_tf2_wyt \
    -p 8022:22 \
    -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
    -v $PWD:/workspace/bert_tf2 -v $PWD/results:/results \
    -v /mnt/dataset2/bert/bert_data/data:/data \
    bert_tf2 $CMD

    # --rm 
    # run 'exit' will direct stop and remove docker container
    # --net=host \   conflict with -p 'WARNING: Published ports are discarded when using host network mode'
    # --shm-size=1g \
    # -v /mnt/dataset2/bert/bert_data/data:/workspace/bert_tf2/data \
    # ln -s /data/ data    # in docker bash
    # Vscode login: ssh root@mlp-milan-02.com -p 8022
