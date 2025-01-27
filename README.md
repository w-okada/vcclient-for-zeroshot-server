VCClient for Zeroshot VC (alpha)
---

# Requirement
## Hardware
- NVIDIA GPU

## Software
- poetry

# Usage

```
git clone https://github.com/w-okada/vcclient-for-zeroshot-server.git
cd vcclient-for-zeroshot-server
cd third_party
git clone https://github.com/Plachtaa/seed-vc.git 
cd seed-vc
git checkout 09d0b5cf131e364e7143c8069d0d03b6889072ba
cd ../..
poetry install
poetry run main start
```
