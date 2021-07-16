set -ex

pip install jina[hub]~=2.0
jina hub push --force $id --secret $secret .
