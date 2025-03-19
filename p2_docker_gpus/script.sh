docker pull --platform linux/amd64 nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

TOKEN=$(curl -s -u '$oauthtoken:API_KEY' "https://nvcr.io/proxy_auth?scope=repository:nvidia/cuda:pull" | jq -r '.token')
curl -L -H "Authorization: Bearer $TOKEN" "https://nvcr.io/v2/nvidia/cuda/manifests/12.8.0-cudnn-devel-ubuntu24.04" | jq .

curl -L -H "Authorization: Bearer $TOKEN" -o top-layer.tar "https://nvcr.io/v2/nvidia/cuda/blobs/sha256:5be3ee9cf7b018b515dc8b38a0d69e271d5d874bc077a616597275265e63af47"
mkdir layer
tar -xf top-layer.tar -C layer

find layer -type f
du -sh layer

docker run --rm -it nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 bash

echo $LD_LIBRARY_PATH

ldconfig -p | grep libcufft
ldconfig -p | grep libcuda